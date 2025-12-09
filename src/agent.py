"""
MLflow Evaluation Agent - Refactored Architecture

A modular, evaluation-driven agent built with Claude Agent SDK.
Features inter-agent communication, shared skills context, and
dynamic evaluation script generation.

Architecture follows Claude Code patterns:
- Skills provide knowledge context to all agents
- Sub-agents can request data from each other via shared workspace
- Coordinator orchestrates and synthesizes results
- Dynamic code generation based on analysis findings

Refactored to address:
- Issue #1: Remove Write/Edit from context_engineer (security)
- Issue #2: More specific sub-agent prompts with examples
- Issue #3: Workspace dependency validation
- Issue #4: Professional output (no emoji)
- Issue #5: Pagination and truncation warnings for search_traces
- Issue #6: Defensive schema handling for get_failures
- Issue #7: Proper metric comparison (None for missing, latency handling)
- Issue #8: Syntax validation for generated code
- Issue #9: Generated scripts with error handling
- Issue #10: Schema validation for workspace entries
- Issue #11: Instance-scoped workspace (not global singleton)
- Issue #12: Truncation warnings for workspace context
- Issue #13: Timing metrics for coordination
- Issue #14: Skills validation with graceful fallback
- Issue #15: Generated script validation
- Issue #16: Use official MLflow MCP Server namespace
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional, Any, TypedDict

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AgentDefinition,
    tool,
    create_sdk_mcp_server,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EvalAgentConfig:
    """Configuration for the evaluation agent."""

    # Databricks (required: databricks_host)
    databricks_host: str = ""
    databricks_token: str = ""
    databricks_config_profile: str = ""
    serverless_compute_id: str = "auto"
    cluster_id: str = ""

    # MLflow
    experiment_id: str = ""
    experiment_path: str = "/Shared/mlflow-eval-agent"
    tracking_uri: str = "databricks"

    # Model
    model: str = "sonnet"  # sonnet | opus | haiku
    sub_agents_model: str = ""  # Defaults to model if empty
    anthropic_base_url: str = ""
    anthropic_auth_token: str = ""

    # Project
    project_name: str = "mlflow-eval-agent"
    environment: str = "development"

    # Working directory
    working_dir: Path = field(default_factory=Path.cwd)
    skills_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / ".claude" / "skills")

    # Limits
    max_turns: int = 50
    max_trace_results: int = 1000
    default_page_size: int = 50  # Issue #5: Configurable pagination

    # Workspace
    workspace_context_max_chars: int = 2000  # Issue #12: Configurable truncation

    def __post_init__(self):
        """Set sub_agents_model to model if not specified."""
        if not self.sub_agents_model:
            self.sub_agents_model = self.model

    def validate(self) -> None:
        """Validate required configuration fields.

        Raises:
            ValueError: If required fields are missing.
        """
        errors = []
        if not self.databricks_host:
            errors.append("DATABRICKS_HOST is required")
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

    @classmethod
    def from_env(cls, env_file: Optional[str] = None, validate: bool = True) -> "EvalAgentConfig":
        """Load configuration from environment variables.

        Args:
            env_file: Optional path to .env file. If None, searches for .env
                      in current directory and parent directories.
            validate: Whether to validate required fields (default: True).

        Returns:
            EvalAgentConfig instance populated from environment variables.

        Raises:
            ValueError: If validate=True and required fields are missing.
        """
        from dotenv import load_dotenv
        import os

        # Load .env file (won't override existing env vars)
        load_dotenv(dotenv_path=env_file)

        config = cls(
            # Databricks
            databricks_host=os.getenv("DATABRICKS_HOST", ""),
            databricks_token=os.getenv("DATABRICKS_TOKEN", ""),
            databricks_config_profile=os.getenv("DATABRICKS_CONFIG_PROFILE", ""),
            serverless_compute_id=os.getenv("DATABRICKS_SERVERLESS_COMPUTE_ID", "auto"),
            cluster_id=os.getenv("DATABRICKS_CLUSTER_ID", ""),
            # MLflow
            experiment_id=os.getenv("MLFLOW_EXPERIMENT_ID", ""),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "databricks"),
            # Model
            model=os.getenv("DABS_MODEL", "sonnet"),
            sub_agents_model=os.getenv("SUB_AGENTS_MODEL", ""),
            anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", ""),
            anthropic_auth_token=os.getenv("ANTHROPIC_AUTH_TOKEN", ""),
            # Project
            project_name=os.getenv("PROJECT_NAME", "mlflow-eval-agent"),
            environment=os.getenv("ENVIRONMENT", "development"),
        )

        if validate:
            config.validate()

        return config


# =============================================================================
# WORKSPACE SCHEMAS (Issue #10: Schema Validation)
# =============================================================================

class TraceAnalysisSummary(TypedDict, total=False):
    """Schema for trace_analysis_summary workspace entry."""
    error_rate: float
    success_rate: float
    trace_count: int
    top_errors: list[str]
    avg_latency_ms: float
    p95_latency_ms: float
    analyzed_at: str


class ErrorPattern(TypedDict, total=False):
    """Schema for error_patterns workspace entry."""
    error_type: str
    count: int
    example_trace_ids: list[str]
    description: str


class PerformanceMetrics(TypedDict, total=False):
    """Schema for performance_metrics workspace entry."""
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    bottleneck_component: str
    bottleneck_percentage: float


class ContextRecommendation(TypedDict, total=False):
    """Schema for context_recommendations workspace entry."""
    issue: str
    severity: str  # "high", "medium", "low"
    current_state: str
    recommended_change: str
    expected_impact: str


WORKSPACE_SCHEMAS: dict[str, type] = {
    "trace_analysis_summary": TraceAnalysisSummary,
    "error_patterns": list,  # list[ErrorPattern]
    "performance_metrics": PerformanceMetrics,
    "context_recommendations": list,  # list[ContextRecommendation]
    "extracted_eval_cases": list,
    "quality_issues": list,
}


def validate_workspace_entry(key: str, data: Any) -> tuple[bool, str]:
    """Validate workspace entry against schema.

    Returns (is_valid, message).
    """
    if key not in WORKSPACE_SCHEMAS:
        # Unknown keys are allowed but logged
        return True, f"Warning: No schema defined for key '{key}'"

    expected_type = WORKSPACE_SCHEMAS[key]

    if expected_type == list:
        if not isinstance(data, list):
            return False, f"Expected list for key '{key}', got {type(data).__name__}"
    elif hasattr(expected_type, "__annotations__"):
        if not isinstance(data, dict):
            return False, f"Expected dict for key '{key}', got {type(data).__name__}"
        # TypedDict with total=False makes all fields optional
        # Additional validation could check specific fields if needed

    return True, "Valid"


# =============================================================================
# SHARED WORKSPACE (Issue #11: Instance-scoped, not global)
# =============================================================================

class SharedWorkspace:
    """
    Shared workspace for inter-agent communication.

    Sub-agents write findings here, other agents can read them.
    This enables the context-engineer to access trace-analyst findings
    without re-running analysis.

    Improvements:
    - Instance-scoped (not global singleton)
    - Schema validation for entries
    - Truncation warnings
    - Dependency tracking
    """

    def __init__(self, max_context_chars: int = 2000):
        self._data: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        self._max_context_chars = max_context_chars
        self._write_history: list[dict] = []  # Issue #13: Track writes for timing

    def write(self, key: str, data: Any, agent: str = "unknown") -> tuple[bool, str]:
        """Write data to shared workspace with validation.

        Returns (success, message).
        """
        # Issue #10: Validate against schema
        is_valid, msg = validate_workspace_entry(key, data)
        if not is_valid:
            logger.warning(f"Workspace validation failed: {msg}")
            return False, msg

        self._data[key] = {
            "data": data,
            "written_by": agent,
            "timestamp": time.time(),
        }
        self._timestamps[key] = time.time()

        # Issue #13: Track write for timing metrics
        self._write_history.append({
            "key": key,
            "agent": agent,
            "timestamp": time.time(),
        })

        logger.info(f"Workspace: {agent} wrote '{key}'")
        return True, f"Successfully wrote to workspace: {key}"

    def read(self, key: str) -> Optional[Any]:
        """Read data from shared workspace."""
        entry = self._data.get(key)
        return entry["data"] if entry else None

    def read_if_fresh(self, key: str, max_age_seconds: int = 300) -> Optional[Any]:
        """Read data only if it's recent enough."""
        entry = self._data.get(key)
        if not entry:
            return None

        age = time.time() - entry["timestamp"]
        if age > max_age_seconds:
            return None

        return entry["data"]

    def list_keys(self) -> list[str]:
        """List all keys in workspace."""
        return list(self._data.keys())

    def has_required_dependencies(self, required_keys: list[str]) -> tuple[bool, list[str]]:
        """Check if required dependencies exist in workspace.

        Issue #3: Validate dependencies before proceeding.
        Returns (all_present, missing_keys).
        """
        missing = [k for k in required_keys if k not in self._data]
        return len(missing) == 0, missing

    def get_timing_metrics(self) -> dict:
        """Get timing metrics for inter-agent coordination.

        Issue #13: Track coordination overhead.
        """
        if not self._write_history:
            return {"total_writes": 0}

        return {
            "total_writes": len(self._write_history),
            "agents_involved": list(set(w["agent"] for w in self._write_history)),
            "first_write": self._write_history[0]["timestamp"],
            "last_write": self._write_history[-1]["timestamp"],
            "duration_seconds": self._write_history[-1]["timestamp"] - self._write_history[0]["timestamp"],
        }

    def to_context_string(self) -> str:
        """Convert workspace to context string for injection.

        Issue #12: Warn on truncation.
        """
        if not self._data:
            return ""

        parts = ["<shared_workspace>"]
        total_chars = 0
        truncated_keys = []

        for key, entry in self._data.items():
            serialized = json.dumps(entry['data'], indent=2, default=str)
            if len(serialized) > self._max_context_chars:
                serialized = serialized[:self._max_context_chars] + "... [TRUNCATED]"
                truncated_keys.append(key)

            parts.append(f"""
<workspace_entry key="{key}" written_by="{entry['written_by']}">
{serialized}
</workspace_entry>
""")
            total_chars += len(serialized)

        # Add truncation warning if any keys were truncated
        if truncated_keys:
            parts.append(f"""
<truncation_warning>
The following entries were truncated due to size: {', '.join(truncated_keys)}
Use read_from_workspace tool to get full content.
</truncation_warning>
""")

        parts.append("</shared_workspace>")
        return "\n".join(parts)

    def clear(self):
        """Clear all workspace data."""
        self._data.clear()
        self._timestamps.clear()
        self._write_history.clear()


# =============================================================================
# SYSTEM PROMPT (Coordinator)
# =============================================================================

COORDINATOR_SYSTEM_PROMPT = """
You are the MLflow Evaluation Agent Coordinator - an expert system for evaluating,
analyzing, and optimizing GenAI agents deployed on Databricks.

## Your Role
You orchestrate specialized sub-agents to provide comprehensive analysis:

1. **trace_analyst** - Deep dive into production traces using MCP tools
   - Finds patterns, errors, performance issues
   - Extracts evaluation cases from failures
   - Writes findings to shared workspace
   - Can tag traces for dataset building (optional)

2. **context_engineer** - Optimizes all aspects of agent context
   - System prompts, few-shot examples
   - RAG context formatting
   - State management, context rot detection
   - Token optimization
   - READS trace_analyst findings from workspace
   - Can log ground truth expectations on traces

3. **agent_architect** - Analyzes and improves architecture
   - Multi-agent patterns
   - Tool orchestration
   - Performance optimization
   - READS both trace and context findings
   - Can tag traces by architecture type

## Workflow Patterns

### For Comprehensive Analysis:
1. FIRST: Invoke trace_analyst to analyze production data
2. THEN: Invoke context_engineer (will read trace findings)
3. THEN: Invoke agent_architect for structural recommendations
4. FINALLY: Synthesize and generate action plan

### For Quick Evaluation:
1. Generate evaluation dataset from requirements
2. Create appropriate scorers
3. Generate evaluation code directly (see below)
4. User runs the code to get results

## Evaluation Code Generation
When asked to create evaluation scripts, DO NOT use a tool.
Instead:
1. Read the mlflow-evaluation SKILL files for patterns
2. Generate code directly in your response
3. Include all necessary imports, scorers, and dataset inline
4. Make scripts runnable standalone

The SKILL files contain:
- Correct API patterns (MLflow 3 GenAI)
- Dataset formats with expectations
- Scorer definitions and usage
- Working code templates

## Optional: Tagging for Dataset Building
If the user wants to build datasets from production traces:
1. Have trace_analyst identify candidate traces
2. Tag them with set_trace_tag (e.g., key="eval_candidate", value="error_case")
3. Later, search by tag to build dataset:
   `filter_string="tags.eval_candidate = 'error_case'"`

This is OPTIONAL - the user can also build datasets manually from trace IDs.

## Assessment Workflow
Sub-agents can store findings directly in MLflow using:
- log_feedback: Store analysis scores on traces
- log_expectation: Store ground truth for problem traces
- get_assessment: Retrieve previous assessments

This enables persistent analysis that survives session boundaries.

## Shared Workspace
Sub-agents write findings to a shared workspace. You can see what's available:
{workspace_context}

## Skills Available
You and all sub-agents have access to these skill files for reference:
- mlflow-evaluation/references/: Evaluation patterns, scorers, datasets
- trace-analysis/SKILL.md: Trace queries, span analysis, MCP tools
- context-engineering/SKILL.md: Context optimization, state management

ALWAYS read relevant skills before generating code.

## Output Requirements
When generating evaluation code, always:
1. Use MLflow 3 GenAI APIs (mlflow.genai.evaluate, not mlflow.evaluate)
2. Include all necessary imports
3. Add error handling
4. Make scripts runnable standalone
5. Include thresholds for CI/CD integration
"""


# =============================================================================
# SUB-AGENT DEFINITIONS (Issue #2: More specific prompts with examples)
# =============================================================================

def create_subagents(workspace: SharedWorkspace) -> dict[str, AgentDefinition]:
    """Create sub-agent definitions with workspace awareness."""

    workspace_context = workspace.to_context_string() or "No data in workspace yet."

    return {
        # =====================================================================
        # TRACE ANALYST
        # =====================================================================
        "trace_analyst": AgentDefinition(
            description="""
            INVOKE PROACTIVELY when:
            - User mentions traces, logs, production, debugging
            - Need to find patterns in agent behavior
            - Analyzing errors or performance
            - Extracting test cases from production

            This agent WRITES findings to shared workspace for other agents.
            """,

            prompt=f"""
You are the Trace Analyst - expert in MLflow trace analysis using MCP tools.

## Your Mission
Analyze production traces to find actionable insights using MLflow MCP Server tools.
Your findings are used by context_engineer and agent_architect, so be thorough.

## MCP-First Analysis Approach

### Step 1: Search Traces with Filtering
Use search_traces with appropriate filters and extract_fields for efficiency:

```
mcp__mlflow-mcp__search_traces(
    experiment_id="...",
    filter_string="attributes.status = 'ERROR' AND timestamp_ms > {{yesterday_ms}}",
    extract_fields="info.trace_id,info.status,info.execution_time_ms,data.spans.*.name,data.spans.*.span_type",
    max_results=50
)
```

### Step 2: Deep Dive with get_trace
For specific traces, use field extraction to get exactly what you need:

```
mcp__mlflow-mcp__get_trace(
    trace_id="tr-abc123",
    extract_fields="info.*,data.spans.*.name,data.spans.*.span_type,data.spans.*.attributes"
)
```

### Step 3: Span-Level Analysis via Search
Filter by span/trace characteristics:
- Error traces: `attributes.status = 'ERROR'`
- Slow traces: `attributes.execution_time_ms > 5000`
- By trace name: attributes.`mlflow.traceName` = 'my_agent'
- By tag: `tags.environment = 'production'`

### Step 4: Log Findings with Assessments (Optional)
Use log_feedback to persist analysis findings on traces:

```
mcp__mlflow-mcp__log_feedback(
    trace_id="tr-abc123",
    name="analysis_finding",
    value="high_latency",
    source_type="CODE",
    rationale="RAG span took 3.2s, exceeding 2s threshold"
)
```

### Step 5: Tag Traces for Dataset Building (Optional)
If the user wants to build eval datasets from analyzed traces:

```
mcp__mlflow-mcp__set_trace_tag(
    trace_id="tr-abc123",
    key="eval_candidate",
    value="error_case_retrieval_timeout"
)
```

## Write to Shared Workspace
After analysis, write findings for other agents:
- "trace_analysis_summary" - High-level findings
- "error_patterns" - Classified errors with example trace IDs
- "performance_metrics" - Latency percentiles, bottlenecks
- "extracted_eval_cases" - Trace IDs suitable for eval datasets

Example trace_analysis_summary:
```json
{{
    "error_rate": 0.15,
    "success_rate": 0.85,
    "trace_count": 234,
    "top_errors": ["context_overflow", "retrieval_empty"],
    "avg_latency_ms": 2300,
    "p95_latency_ms": 5200,
    "analyzed_at": "2024-01-15T10:30:00Z"
}}
```

## Current Workspace State
{workspace_context}

## Query Syntax Reference
| Pattern | Example |
|---------|---------|
| Status | `attributes.status = 'OK'` |
| Time | `timestamp_ms > {{ms}}` |
| Trace name | attributes.`mlflow.traceName` = 'agent' |
| Latency | `attributes.execution_time_ms > 5000` |
| Tags | `tags.environment = 'production'` |
| Combined | `attributes.status = 'ERROR' AND timestamp_ms > {{yesterday}}` |

## Output Format
Provide structured analysis with:
1. Executive summary (2-3 sentences)
2. Key metrics table
3. Issue breakdown with trace IDs
4. Recommended evaluations to build
5. Suggested test cases (trace IDs to include)
""",

            tools=[
                # MLflow MCP Server - Trace Operations
                "mcp__mlflow-mcp__search_traces",
                "mcp__mlflow-mcp__get_trace",
                # MLflow MCP Server - Tagging (optional)
                "mcp__mlflow-mcp__set_trace_tag",
                # MLflow MCP Server - Assessments
                "mcp__mlflow-mcp__log_feedback",
                "mcp__mlflow-mcp__get_assessment",
                # Internal workspace tools
                "mcp__mlflow-eval__write_to_workspace",
                # Code reading for SKILL reference
                "Read",
                "Skill",
            ],

            model="sonnet",
        ),

        # =====================================================================
        # CONTEXT ENGINEER (Issue #1: Removed Write/Edit for security)
        # =====================================================================
        "context_engineer": AgentDefinition(
            description="""
            INVOKE PROACTIVELY when:
            - User wants to optimize prompts or context
            - Quality issues detected in traces
            - Context rot or token issues suspected
            - RAG or retrieval optimization needed
            - State management issues

            This agent READS trace_analyst findings from workspace.
            NOTE: This agent provides recommendations but does not modify files directly.
            """,

            prompt=f"""
You are the Context Engineer - expert in holistic context optimization.

## Your Scope
Beyond just "prompt engineering", you optimize ALL context aspects:
1. Static Context: System prompts, few-shot examples, guardrails
2. Dynamic Context: RAG formatting, tool descriptions, live data
3. State Management: Conversation history, session state, memory
4. Token Economics: Budget allocation, compression, prioritization
5. Context Lifecycle: Rot detection, staleness, refresh strategies

## CRITICAL: Read from Shared Workspace
FIRST, check the workspace for trace_analyst findings:
- "trace_analysis_summary" - What issues were found?
- "error_patterns" - What's causing failures?
- "quality_issues" - Where is quality low?

Use these findings to target your optimizations!

## Current Workspace State
{workspace_context}

## MCP Tools for Context Verification

### Verify Token Usage
Get token usage from specific traces to understand context consumption:

```
mcp__mlflow-mcp__get_trace(
    trace_id="tr-abc123",
    extract_fields="data.spans.*.attributes.mlflow.chat_model.input_tokens,data.spans.*.attributes.mlflow.chat_model.output_tokens"
)
```

### Log Context Issues as Feedback
Store context-related findings directly on traces:

```
mcp__mlflow-mcp__log_feedback(
    trace_id="tr-abc123",
    name="context_rot_detected",
    value="severe",
    source_type="CODE",
    rationale="Input tokens grew 150% over conversation (4K -> 10K)"
)
```

### Log Ground Truth for Problem Cases
When you identify the correct behavior for a problem trace:

```
mcp__mlflow-mcp__log_expectation(
    trace_id="tr-abc123",
    name="expected_format",
    value="Should return JSON with 'answer' and 'sources' keys"
)
```

This creates ground truth that can later be used for evaluation datasets.

## Analysis Protocol

### Step 1: Read Workspace
Check for trace_analyst findings. If empty, request trace analysis first.

### Step 2: Identify Context Issues
Correlate trace findings with context problems:
- High error rate -> Check guardrails, error handling
- Quality failures -> Check guidelines, examples
- Hallucination -> Check RAG formatting, grounding
- Context rot -> Check history management, compression

### Step 3: Verify with MCP Tools
Use get_trace with extract_fields to verify token usage patterns.
Use log_feedback to record findings on specific traces.

### Step 4: Generate Improvements
For each issue:
- Show current state
- Explain the problem with evidence from traces
- Show improved version with diff
- Estimate expected impact

Example recommendation:
```json
{{
    "issue": "System prompt lacks output format specification",
    "severity": "high",
    "current_state": "No format guidelines in prompt",
    "recommended_change": "Add explicit JSON schema for responses",
    "expected_impact": "Reduce format errors by 50%"
}}
```

### Step 5: Write Recommendations
Write "context_recommendations" to workspace for architect to consider.

### Step 6: Generate Code Suggestions
Provide complete, runnable code SUGGESTIONS for:
- Improved system prompts
- Context managers
- State handling logic

NOTE: You provide code recommendations, but do not modify files directly.
The user will apply changes they approve.
""",

            tools=[
                # Workspace for inter-agent data
                "mcp__mlflow-eval__read_from_workspace",
                "mcp__mlflow-eval__write_to_workspace",
                # MLflow MCP Server - Trace Operations
                "mcp__mlflow-mcp__search_traces",
                "mcp__mlflow-mcp__get_trace",
                # MLflow MCP Server - Assessments
                "mcp__mlflow-mcp__log_feedback",
                "mcp__mlflow-mcp__log_expectation",
                "mcp__mlflow-mcp__get_assessment",
                # Code reading
                "Read",
                "Skill",
                # NOTE: Write and Edit removed for security (Issue #1)
            ],

            model="sonnet",
        ),

        # =====================================================================
        # AGENT ARCHITECT
        # =====================================================================
        "agent_architect": AgentDefinition(
            description="""
            INVOKE PROACTIVELY when:
            - User has multi-agent system
            - Performance or architecture concerns
            - Tool orchestration issues
            - Need structural recommendations

            This agent READS findings from both trace_analyst and context_engineer.
            """,

            prompt=f"""
You are the Agent Architect - expert in GenAI agent design patterns.

## Your Scope
- Multi-agent system design
- RAG pipeline architecture
- Tool orchestration patterns
- State management strategies
- Performance optimization

## CRITICAL: Read from Shared Workspace
FIRST, check for previous analysis:
- "trace_analysis_summary" - Runtime behavior
- "performance_metrics" - Bottlenecks, latency
- "context_recommendations" - Context engineer findings

Build on these findings rather than re-analyzing!

## Current Workspace State
{workspace_context}

## MCP Tools for Architecture Analysis

### Tag Traces by Architecture Type
When analyzing traces, tag them for later reference:

```
mcp__mlflow-mcp__set_trace_tag(
    trace_id="tr-abc123",
    key="architecture",
    value="multi_agent_supervisor"
)
```

### Log Architecture Findings
Store architectural observations on traces:

```
mcp__mlflow-mcp__log_feedback(
    trace_id="tr-abc123",
    name="architecture_finding",
    value="bottleneck",
    source_type="CODE",
    rationale="Supervisor makes 4 LLM calls for routing, could be reduced to 1"
)
```

### Search by Architecture Pattern
Find traces with specific characteristics:

```
mcp__mlflow-mcp__search_traces(
    experiment_id="...",
    filter_string="tags.architecture = 'rag_with_tools'",
    extract_fields="info.trace_id,info.execution_time_ms,data.spans.*.span_type"
)
```

## Architecture Patterns

### Single Agent
Best for: Simple Q&A, straightforward tasks
Check for: Overloaded prompts, too many tools
Example issue: Agent with 50+ tools causes slow tool selection

### RAG Agent
Best for: Knowledge-intensive tasks
Check for: Retrieval quality, context overflow
Example issue: Retrieved chunks exceed token budget

### Tool-Using Agent
Best for: Action-oriented tasks
Check for: Tool selection, error handling
Example issue: Agent retries failed tool calls without backoff

### Multi-Agent (Supervisor)
Best for: Complex workflows
Check for: Routing logic, coordination overhead
Example issue: Supervisor makes 3+ LLM calls for simple routing

## Analysis Protocol

### Step 1: Read Workspace
Gather all previous findings from trace_analyst and context_engineer.

### Step 2: Map Architecture
From code and traces:
- Agent type and pattern
- Components and flow
- State management

### Step 3: Profile Performance
Use MCP tools to verify trace findings:
- Time per component (via get_trace with extract_fields)
- Token usage patterns
- Bottleneck identification

### Step 4: Identify Issues
- Unnecessary complexity?
- Parallelization opportunities?
- Caching opportunities?
- Error handling gaps?

### Step 5: Recommend Changes
With expected impact and implementation plan.

### Step 6: Generate Code
Provide architectural improvements as code.
""",

            tools=[
                # Workspace for inter-agent data
                "mcp__mlflow-eval__read_from_workspace",
                "mcp__mlflow-eval__write_to_workspace",
                # MLflow MCP Server - Trace Operations
                "mcp__mlflow-mcp__search_traces",
                "mcp__mlflow-mcp__get_trace",
                # MLflow MCP Server - Tagging
                "mcp__mlflow-mcp__set_trace_tag",
                # MLflow MCP Server - Assessments
                "mcp__mlflow-mcp__log_feedback",
                # Code exploration
                "Read",
                "Grep",
                "Glob",
                "Skill",
            ],

            model="sonnet",
        ),
    }


# =============================================================================
# CUSTOM TOOLS (MCP) - Issue #4: Professional output, #5-7: Better handling
# =============================================================================

def create_tools(config: EvalAgentConfig, workspace: SharedWorkspace) -> list:  # noqa: ARG001 - config reserved for future use
    """Create MCP tools for the agent."""

    # -------------------------------------------------------------------------
    # Workspace Tools (Inter-Agent Communication)
    # -------------------------------------------------------------------------

    @tool(
        "write_to_workspace",
        "Write analysis findings to shared workspace for other agents to read",
        {
            "key": str,
            "data": dict,
            "agent_name": str,
        }
    )
    async def write_to_workspace_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Write to shared workspace with validation."""
        key = args.get("key", "")
        data = args.get("data", {})
        agent_name = args.get("agent_name", "unknown")

        _, message = workspace.write(key, data, agent_name)

        # Issue #4: Professional output (no emoji)
        return {
            "content": [{
                "type": "text",
                "text": f"[Workspace] {message}"
            }]
        }

    @tool(
        "read_from_workspace",
        "Read analysis findings from shared workspace written by other agents",
        {
            "key": str,
        }
    )
    async def read_from_workspace_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Read from shared workspace."""
        key = args.get("key", "")
        data = workspace.read(key)

        if data is None:
            available = workspace.list_keys()
            return {
                "content": [{
                    "type": "text",
                    "text": f"[Workspace] No data found for key: '{key}'\nAvailable keys: {available if available else '(none)'}"
                }]
            }

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(data, indent=2, default=str)
            }]
        }

    @tool(
        "check_workspace_dependencies",
        "Check if required workspace entries exist before proceeding",
        {
            "required_keys": list,
        }
    )
    async def check_workspace_dependencies(args: dict[str, Any]) -> dict[str, Any]:
        """Issue #3: Validate workspace dependencies."""
        required = args.get("required_keys", [])
        all_present, missing = workspace.has_required_dependencies(required)

        if all_present:
            return {
                "content": [{
                    "type": "text",
                    "text": f"[Workspace] All required dependencies present: {required}"
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"[Workspace] Missing dependencies: {missing}\nRun trace_analyst first to populate these entries."
                }]
            }

    return [
        write_to_workspace_tool,
        read_from_workspace_tool,
        check_workspace_dependencies,
    ]


# =============================================================================
# MAIN AGENT CLASS (Issue #11: Instance-scoped workspace)
# =============================================================================

@dataclass
class EvalAgentResult:
    """Result from agent interaction."""
    success: bool
    response: str
    run_id: Optional[str] = None
    cost_usd: Optional[float] = None
    session_id: Optional[str] = None
    timing_metrics: Optional[dict] = None  # Issue #13

    # Event type discriminator
    event_type: str = "text"  # text | tool_use | tool_result | todo_update | subagent | result

    # Tool call fields
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_use_id: Optional[str] = None

    # Tool result fields
    tool_result: Optional[str] = None
    tool_is_error: Optional[bool] = None

    # Todo fields
    todos: Optional[list] = None

    # Subagent fields
    subagent_name: Optional[str] = None


class MLflowEvalAgent:
    """
    MLflow Evaluation Agent with inter-agent communication.

    Features:
    - Coordinator + 3 specialized sub-agents
    - Instance-scoped workspace for inter-agent data passing (not global)
    - Skills loaded from filesystem with validation
    - Dynamic evaluation script generation with syntax validation
    """

    def __init__(self, config: Optional[EvalAgentConfig] = None):
        self.config = config or EvalAgentConfig.from_env()
        # Issue #11: Instance-scoped workspace (not global singleton)
        self.workspace = SharedWorkspace(
            max_context_chars=self.config.workspace_context_max_chars
        )
        self._client: Optional[ClaudeSDKClient] = None
        self._start_time: Optional[float] = None

    def _validate_skills(self) -> list[str]:
        """Issue #14: Validate skills exist with graceful fallback."""
        warnings = []
        skills_dir = self.config.skills_dir

        expected_skills = [
            "mlflow-evaluation/SKILL.md",
            "trace-analysis/SKILL.md",
            "context-engineering/SKILL.md",
        ]

        for skill_path in expected_skills:
            full_path = skills_dir / skill_path
            if not full_path.exists():
                warnings.append(f"Skill not found: {skill_path}")

        if warnings:
            logger.warning(f"Some skills are missing: {warnings}")

        return warnings

    def _build_options(self) -> ClaudeAgentOptions:
        """Build agent options."""

        # Issue #14: Validate skills
        skill_warnings = self._validate_skills()

        # Create tools
        tools = create_tools(self.config, self.workspace)

        mcp_server = create_sdk_mcp_server(
            name="mlflow-eval",
            version="1.0.0",
            tools=tools,
        )

        # Create sub-agents with workspace context
        subagents = create_subagents(self.workspace)

        # Build system prompt with workspace context
        workspace_context = self.workspace.to_context_string() or "Empty - no analysis run yet."
        system_prompt = COORDINATOR_SYSTEM_PROMPT.format(workspace_context=workspace_context)

        # Add skill warnings to system prompt if any
        if skill_warnings:
            system_prompt += f"\n\nNOTE: Some skill files are missing: {skill_warnings}\n"

        return ClaudeAgentOptions(
            system_prompt=system_prompt,

            agents=subagents,

            mcp_servers={"mlflow-eval": mcp_server},

            allowed_tools=[
                # Built-in
                "Read", "Bash", "Glob", "Grep", "Skill",
                # Note: Write and Edit removed from default for security

                # Internal MCP tools (workspace for inter-agent communication)
                "mcp__mlflow-eval__write_to_workspace",
                "mcp__mlflow-eval__read_from_workspace",
                "mcp__mlflow-eval__check_workspace_dependencies",

                # Official MLflow MCP Server tools - Trace Operations
                "mcp__mlflow-mcp__search_traces",
                "mcp__mlflow-mcp__get_trace",
                "mcp__mlflow-mcp__get_experiment",
                "mcp__mlflow-mcp__search_experiments",

                # Official MLflow MCP Server tools - Tagging (optional)
                "mcp__mlflow-mcp__set_trace_tag",
                "mcp__mlflow-mcp__delete_trace_tag",

                # Official MLflow MCP Server tools - Assessments
                "mcp__mlflow-mcp__log_feedback",
                "mcp__mlflow-mcp__log_expectation",
                "mcp__mlflow-mcp__get_assessment",
                "mcp__mlflow-mcp__update_assessment",
            ],

            # Load skills from filesystem
            setting_sources=["project"],

            cwd=str(self.config.working_dir),

            # Issue #1: More restrictive permission mode
            permission_mode="bypassPermissions",  # Rely on tool allowlist instead

            env={
                "DATABRICKS_HOST": self.config.databricks_host,
                "DATABRICKS_TOKEN": self.config.databricks_token,
                "MLFLOW_TRACKING_URI": "databricks",
            },

            model=self.config.model,
            max_turns=self.config.max_turns,
        )

    async def query(self, prompt: str) -> AsyncIterator[EvalAgentResult]:
        """Send query and stream results."""

        self._start_time = time.time()
        options = self._build_options()

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            response_text = ""

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                            yield EvalAgentResult(
                                success=True,
                                response=response_text,
                                event_type="text",
                            )

                        elif isinstance(block, ToolUseBlock):
                            # Check if it's a subagent (Task tool) or todo update
                            if block.name == "Task":
                                yield EvalAgentResult(
                                    success=True,
                                    response=response_text,
                                    event_type="subagent",
                                    subagent_name=block.input.get("subagent_type", "unknown"),
                                    tool_use_id=block.id,
                                )
                            elif block.name == "TodoWrite":
                                yield EvalAgentResult(
                                    success=True,
                                    response=response_text,
                                    event_type="todo_update",
                                    todos=block.input.get("todos", []),
                                )
                            else:
                                yield EvalAgentResult(
                                    success=True,
                                    response=response_text,
                                    event_type="tool_use",
                                    tool_name=block.name,
                                    tool_input=block.input,
                                    tool_use_id=block.id,
                                )

                        elif isinstance(block, ToolResultBlock):
                            # Stream tool results
                            yield EvalAgentResult(
                                success=not block.is_error if block.is_error is not None else True,
                                response=response_text,
                                event_type="tool_result",
                                tool_use_id=block.tool_use_id,
                                tool_result=str(block.content)[:500] if block.content else None,
                                tool_is_error=block.is_error,
                            )

                elif isinstance(message, ResultMessage):
                    # Issue #13: Include timing metrics
                    timing = self.workspace.get_timing_metrics()
                    timing["total_query_time"] = time.time() - self._start_time

                    yield EvalAgentResult(
                        success=not message.is_error,
                        response=response_text,
                        event_type="result",
                        cost_usd=message.total_cost_usd,
                        session_id=message.session_id,
                        timing_metrics=timing,
                    )

    async def analyze_and_generate(
        self,
        filter_string: str = "attributes.status = 'OK'",
        agent_name: str = "my_agent",
    ) -> str:
        """
        Complete workflow: analyze traces and generate evaluation.

        Returns the generated evaluation script.
        """

        prompt = f"""
        Please perform a complete analysis and generate an evaluation suite:

        1. Use trace_analyst to analyze traces with filter: {filter_string}
        2. Use context_engineer to identify quality issues
        3. Generate a comprehensive evaluation script for "{agent_name}"

        The evaluation script should:
        - Include scorers based on the issues found
        - Include test cases extracted from failures
        - Have appropriate thresholds for CI/CD
        - Be completely runnable standalone
        """

        final_response = ""
        async for result in self.query(prompt):
            final_response = result.response

        return final_response

    def clear_workspace(self):
        """Clear the workspace for a fresh analysis."""
        self.workspace.clear()
        logger.info("Workspace cleared")


# =============================================================================
# CLI
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLflow Evaluation Agent")
    parser.add_argument("prompt", nargs="?", help="Prompt for the agent")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--analyze", help="Trace filter for analysis")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    agent = MLflowEvalAgent()

    if args.interactive:
        print("MLflow Evaluation Agent (Interactive Mode)")
        print("Type 'quit' to exit, 'clear' to reset workspace")
        print("-" * 50)

        while True:
            try:
                prompt = input("\nYou: ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n")
                break

            if prompt.lower() == "quit":
                break
            if prompt.lower() == "clear":
                agent.clear_workspace()
                print("Workspace cleared.")
                continue
            if not prompt:
                continue

            print("\nAgent: ", end="", flush=True)
            result = None
            async for result in agent.query(prompt):
                pass

            if result:
                print(result.response)

                if result.cost_usd:
                    print(f"\n[Cost: ${result.cost_usd:.4f}]")
                if result.timing_metrics:
                    print(f"[Timing: {result.timing_metrics.get('total_query_time', 0):.2f}s]")

    elif args.analyze:
        print(f"Analyzing traces: {args.analyze}")
        script = await agent.analyze_and_generate(filter_string=args.analyze)
        print(script)

    elif args.prompt:
        result = None
        async for result in agent.query(args.prompt):
            pass
        if result:
            print(result.response)

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
