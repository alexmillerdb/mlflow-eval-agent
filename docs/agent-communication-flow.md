# Agent Communication Flow

This diagram shows how the MLflow Evaluation Agent orchestrates sub-agents through a shared workspace pattern.

## Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Coordinator as Coordinator Agent
    participant Task as Task Tool
    participant TA as trace_analyst
    participant CE as context_engineer
    participant AA as agent_architect
    participant WS as Shared Workspace
    participant MCP as MLflow MCP Server
    participant Skills as Skills Layer

    User->>Coordinator: "Analyze my RAG agent's traces"
    Coordinator->>Coordinator: Build options & load skills
    Coordinator->>Skills: Load mlflow-evaluation, trace-analysis skills
    Skills-->>Coordinator: Skill patterns loaded

    Note over Coordinator: Phase 1: Trace Analysis
    Coordinator->>Task: Delegate trace analysis
    Task->>TA: Analyze production traces
    TA->>MCP: search_traces(filter_string)
    MCP-->>TA: Return trace results
    TA->>MCP: get_trace(trace_id) [for each]
    MCP-->>TA: Return trace details
    TA->>TA: Classify errors, profile latency
    TA->>WS: write("trace_analysis_summary", findings)
    TA->>WS: write("error_patterns", errors)
    TA->>WS: write("performance_metrics", metrics)
    TA-->>Task: Analysis complete
    Task-->>Coordinator: trace_analyst results

    Note over Coordinator: Phase 2: Context Engineering
    Coordinator->>Task: Delegate context optimization
    Task->>CE: Identify quality issues
    CE->>WS: read("trace_analysis_summary")
    WS-->>CE: Return trace findings
    CE->>Skills: Load context-engineering patterns
    Skills-->>CE: Context optimization patterns
    CE->>CE: Analyze context rot, token usage
    CE->>WS: write("context_recommendations", recommendations)
    CE-->>Task: Recommendations ready
    Task-->>Coordinator: context_engineer results

    Note over Coordinator: Phase 3: Architecture Analysis
    Coordinator->>Task: Delegate architecture review
    Task->>AA: Analyze agent structure
    AA->>WS: read("trace_analysis_summary")
    AA->>WS: read("context_recommendations")
    WS-->>AA: Return all findings
    AA->>MCP: set_trace_tag(trace_id, "architecture_issue")
    MCP-->>AA: Tag set
    AA-->>Task: Architecture insights
    Task-->>Coordinator: agent_architect results

    Note over Coordinator: Phase 4: Synthesis & Output
    Coordinator->>WS: read all workspace data
    WS-->>Coordinator: Complete findings
    Coordinator->>Skills: Load eval script templates
    Skills-->>Coordinator: Scorer patterns, dataset formats
    Coordinator->>Coordinator: Generate evaluation script
    Coordinator-->>User: Evaluation suite + recommendations
```

## Key Concepts

### 1. Sequential Sub-agent Execution
Sub-agents execute in a specific order where each builds on previous findings:
1. **trace_analyst** - Analyzes production traces first
2. **context_engineer** - Reads trace findings, generates optimizations
3. **agent_architect** - Reads all findings, provides structural recommendations

### 2. Workspace-based Communication
Sub-agents communicate through a shared workspace rather than direct message passing:
- `trace_analyst` writes: `trace_analysis_summary`, `error_patterns`, `performance_metrics`
- `context_engineer` writes: `context_recommendations`
- All data persists for the session and is accessible to later agents

### 3. MCP Tool Integration
The MLflow MCP Server provides trace operations:
- `search_traces` - Query traces with filters
- `get_trace` - Get detailed trace by ID
- `set_trace_tag` - Tag traces for categorization

### 4. Skills Layer
Skills provide code patterns and domain knowledge:
- `mlflow-evaluation` - Scorer patterns, dataset formats
- `trace-analysis` - Query syntax, error classification
- `context-engineering` - Prompt optimization, token budgets

## Generating PNG/SVG

To generate an image from the `.mmd` file:

```bash
# Using Mermaid CLI (npm install -g @mermaid-js/mermaid-cli)
mmdc -i agent-communication-flow.mmd -o agent-communication-flow.png

# Or for SVG
mmdc -i agent-communication-flow.mmd -o agent-communication-flow.svg
```
