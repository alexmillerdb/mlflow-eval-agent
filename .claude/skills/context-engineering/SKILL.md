---
name: context-engineering
description: Holistic context optimization for GenAI agents beyond prompt engineering. Use when optimizing system prompts, few-shot examples, RAG context formatting, state management, token budgets, or detecting context rot. Also use when context analysis from traces reveals high token usage, context overflow, or quality degradation over conversation turns. Designed for sub-agent use.
---

# Context Engineering

Comprehensive context optimization covering system prompts, dynamic context, state management, and token economics.

## Sub-Agent Design

This skill is for a context engineering sub-agent that:
1. Reads trace analysis from workspace (requires trace_analyst findings)
2. Analyzes context issues based on trace patterns
3. Generates targeted optimization recommendations
4. Writes recommendations to workspace for architect review

## Context Engineering Framework

### 1. Static Context Optimization

**System Prompt Architecture**

```python
def analyze_system_prompt(prompt: str) -> dict:
    """Analyze system prompt for optimization opportunities."""
    lines = prompt.strip().split('\n')
    sections = identify_sections(prompt)

    issues = []

    # Check length
    if len(prompt) > 4000:
        issues.append({
            "type": "prompt_too_long",
            "current_chars": len(prompt),
            "recommendation": "Consider splitting into base + conditional sections"
        })

    # Check for redundancy
    if has_repeated_instructions(prompt):
        issues.append({
            "type": "redundant_instructions",
            "recommendation": "Consolidate repeated instructions"
        })

    # Check for implicit vs explicit
    if count_examples(prompt) > 5:
        issues.append({
            "type": "excessive_examples",
            "count": count_examples(prompt),
            "recommendation": "Reduce to 2-3 diverse examples, let model generalize"
        })

    return {
        "char_count": len(prompt),
        "estimated_tokens": len(prompt) // 4,
        "section_count": len(sections),
        "issues": issues
    }
```

**Few-Shot Example Design**

```markdown
## Few-Shot Best Practices

1. **Diversity over quantity**: 2-3 diverse examples > 10 similar ones
2. **Edge cases**: Include boundary conditions, not just happy path
3. **Format demonstration**: Show exact output format you expect
4. **Negative examples**: Show what NOT to do when common mistakes exist

Example structure:
<example>
<input>User query here</input>
<reasoning>Brief thought process (optional)</reasoning>
<output>Exact format you want</output>
</example>
```

### 2. Dynamic Context Optimization

**RAG Context Formatting**

```python
def optimize_rag_context(retrieved_chunks: list, max_tokens: int = 4000) -> str:
    """Format RAG context for optimal LLM consumption."""

    # Limit chunks (quality > quantity)
    chunks = retrieved_chunks[:5]  # Top 5 is usually sufficient

    # Format for clarity
    formatted_parts = []
    for i, chunk in enumerate(chunks, 1):
        # Truncate very long chunks
        content = chunk["content"][:1000] if len(chunk["content"]) > 1000 else chunk["content"]

        formatted_parts.append(f"""
<source id="{i}" relevance="{chunk.get('score', 'N/A'):.2f}">
{content}
</source>
""")

    context = "\n".join(formatted_parts)

    # Add grounding instruction
    return f"""
<retrieved_context>
Use ONLY the following sources to answer. Cite sources by ID.
{context}
</retrieved_context>
"""
```

**Tool Result Summarization**

```python
def summarize_tool_result(result: dict, max_chars: int = 500) -> str:
    """Summarize verbose tool output for context efficiency."""

    # Extract key fields only
    essential_fields = ["name", "value", "status", "error", "summary"]
    summary = {k: v for k, v in result.items() if k in essential_fields}

    # Truncate long values
    for k, v in summary.items():
        if isinstance(v, str) and len(v) > 200:
            summary[k] = v[:197] + "..."
        elif isinstance(v, list) and len(v) > 5:
            summary[k] = v[:5] + [f"... and {len(v)-5} more"]

    return json.dumps(summary, indent=2)
```

### 3. State Management Patterns

**Structured State vs Message History**

```python
# PROBLEM: Message history grows unbounded
messages = [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "content": "... 5000 tokens of JSON ..."},
    # ... continues growing
]

# SOLUTION: Structured state
state = {
    "topic": "quarterly report analysis",
    "entities": ["Company A", "Company B"],
    "filters": {"quarter": "Q3 2024", "metric": "revenue"},
    "key_findings": [
        "Revenue up 15%",
        "Market share stable"
    ],
    "turn_count": 7
}
# Fixed size, semantic compression
```

**State Schema for Multi-Turn**

```python
class ConversationState(TypedDict):
    """Structured state for multi-turn conversations."""
    topic: str                    # Current topic focus
    entities: list[str]           # Entities discussed
    filters: dict                 # Active filters/constraints
    key_findings: list[str]       # Important conclusions (max 10)
    pending_questions: list[str]  # Unresolved questions
    turn_count: int               # Conversation length

def compress_to_state(messages: list) -> ConversationState:
    """Extract structured state from message history."""
    # Use LLM to extract key information
    extraction_prompt = """
    Extract from this conversation:
    1. Main topic
    2. Key entities mentioned
    3. Any filters or constraints applied
    4. Important findings or conclusions
    5. Unanswered questions

    Conversation:
    {messages}
    """
    # ... LLM call to extract ...
```

### 4. Token Budget Management

**Budget Allocation Strategy**

```python
def allocate_token_budget(
    total_budget: int = 128000,
    task_type: str = "rag"
) -> dict:
    """Allocate context window across components."""

    allocations = {
        "rag": {
            "system_prompt": 0.10,     # 12.8K
            "retrieved_context": 0.35, # 44.8K
            "conversation_history": 0.20, # 25.6K
            "current_query": 0.05,     # 6.4K
            "response_buffer": 0.30    # 38.4K for output
        },
        "tool_calling": {
            "system_prompt": 0.15,
            "tool_descriptions": 0.10,
            "conversation_history": 0.25,
            "tool_results": 0.20,
            "response_buffer": 0.30
        },
        "multi_agent": {
            "system_prompt": 0.10,
            "agent_context": 0.20,
            "shared_state": 0.15,
            "current_task": 0.15,
            "response_buffer": 0.40
        }
    }

    allocation = allocations.get(task_type, allocations["rag"])
    return {k: int(v * total_budget) for k, v in allocation.items()}
```

### 5. Context Rot Detection

**Symptoms and Detection**

```python
def detect_context_rot(traces: list) -> dict:
    """Detect context rot from trace patterns."""

    # Track metrics across conversation turns
    turn_metrics = []
    for trace in sorted(traces, key=lambda t: t.info.timestamp_ms):
        llm_spans = [s for s in trace.data.spans
                     if s.span_type in ["LLM", "CHAT_MODEL"]]

        total_input = sum(
            s.attributes.get("mlflow.chat_model.input_tokens", 0)
            for s in llm_spans
        )
        total_latency = sum(
            (s.end_time_ns - s.start_time_ns) / 1e6
            for s in llm_spans
        )

        turn_metrics.append({
            "input_tokens": total_input,
            "latency_ms": total_latency
        })

    # Detect growth patterns
    if len(turn_metrics) < 3:
        return {"has_context_rot": False, "reason": "insufficient_data"}

    first_tokens = turn_metrics[0]["input_tokens"]
    last_tokens = turn_metrics[-1]["input_tokens"]
    first_latency = turn_metrics[0]["latency_ms"]
    last_latency = turn_metrics[-1]["latency_ms"]

    token_growth = (last_tokens - first_tokens) / first_tokens * 100 if first_tokens else 0
    latency_growth = (last_latency - first_latency) / first_latency * 100 if first_latency else 0

    return {
        "has_context_rot": token_growth > 50 or latency_growth > 80,
        "token_growth_pct": round(token_growth, 1),
        "latency_growth_pct": round(latency_growth, 1),
        "turns_analyzed": len(turn_metrics),
        "recommendation": "Implement sliding window or semantic compression" if token_growth > 50 else None
    }
```

### 6. Compression Strategies

**Tier 1: Sliding Window**
```python
def sliding_window(messages: list, max_messages: int = 10) -> list:
    """Keep only recent messages."""
    return messages[-max_messages:]
```

**Tier 2: Selective Retention**
```python
def selective_retain(messages: list) -> list:
    """Keep system, recent, and important messages."""
    result = []

    # Always keep system message
    result.extend([m for m in messages if m["role"] == "system"])

    # Keep last N user/assistant
    user_assistant = [m for m in messages if m["role"] in ["user", "assistant"]]
    result.extend(user_assistant[-6:])  # Last 3 turns

    # Summarize old tool results
    for m in messages:
        if m["role"] == "tool" and m not in result:
            if len(m["content"]) > 500:
                m["content"] = summarize_tool_result(json.loads(m["content"]))
            result.append(m)

    return sorted(result, key=lambda m: messages.index(m))
```

**Tier 3: Semantic Compression**
```python
def semantic_compress(messages: list, keep_recent: int = 5) -> list:
    """Use LLM to summarize old context."""
    recent = messages[-keep_recent:]
    old = messages[:-keep_recent]

    if not old:
        return messages

    # Generate summary of old messages
    summary = llm_summarize(old)

    return [
        {"role": "system", "content": f"[Prior context summary: {summary}]"},
        *recent
    ]
```

## MCP Tools for Context Verification

Use MLflow MCP Server tools to verify context issues during analysis.

### Verify Token Usage

Get token usage from specific traces to understand context consumption:

```
mcp__mlflow-eval__get_trace(
    trace_id="tr-abc123",
    extract_fields="data.spans.*.attributes.mlflow.chat_model.input_tokens,data.spans.*.attributes.mlflow.chat_model.output_tokens"
)
```

### Log Context Issues as Feedback

Store context-related findings directly on traces (persists in MLflow):

```
mcp__mlflow-eval__log_feedback(
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
mcp__mlflow-eval__log_expectation(
    trace_id="tr-abc123",
    name="expected_format",
    value="Should return JSON with 'answer' and 'sources' keys"
)
```

This creates ground truth that can later be used for evaluation datasets.

### Search for Context Issues

Find traces with high token usage or long latency:

```
mcp__mlflow-eval__search_traces(
    experiment_id="123",
    filter_string="attributes.execution_time_ms > 10000",
    extract_fields="info.trace_id,info.execution_time_ms,data.spans.*.attributes.mlflow.chat_model.input_tokens"
)
```

## Optimization Workflow

1. **Read workspace** for trace analysis findings
2. **Verify with MCP** - Use get_trace to examine specific problem traces
3. **Identify issues** by correlating trace patterns with context problems:
   - High error rate → Check guardrails, format specs
   - Quality failures → Check guidelines, examples
   - Hallucination → Check RAG grounding
   - Latency growth → Check context rot
   - Token bloat → Check tool result handling

4. **Log findings** - Use log_feedback to persist findings on specific traces
5. **Generate recommendations** with severity and impact:
```python
{
    "issue": "System prompt lacks output format specification",
    "severity": "high",
    "evidence": "follows_instructions score: 37.5%",
    "current_state": "No format guidelines in prompt",
    "recommended_change": "Add explicit JSON schema for responses",
    "expected_impact": "Improve follows_instructions by 20-40%"
}
```

6. **Write to workspace** for architect review

## When to Use MCP vs Python SDK

| Use Case | Approach |
|----------|----------|
| Exploring traces for issues | MCP tools |
| Logging context findings | MCP tools |
| Generating eval scripts | Python SDK |
| Building context optimization code | Python SDK |

## Detailed Reference

For complete optimization strategies, see:
- [patterns-context-optimization.md](../mlflow-evaluation/references/patterns-context-optimization.md)
