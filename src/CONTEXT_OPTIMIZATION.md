# Agent Context Optimization Analysis

Analysis of trace `tr-c9426fa0a3e2a430688905a4ba8d66a1` to understand and fix session performance issues.

## Problem Summary

**session_5** (validation task) took **236 seconds with 21 LLM calls** due to linear context growth without pruning.

### Key Metrics

| Metric | Value | Impact |
|--------|-------|--------|
| Session Duration | 236 seconds (4 min) | 28% of total trace time |
| LLM Calls | 21 calls | High API overhead |
| Context Growth | 10→53 messages | +2 messages per tool use |
| Input Data | 74KB→159KB per call | 117KB average |
| Total Input | 2.5MB across session | Massive token costs |
| Cost | $1.29 for one session | Expensive iteration |

### Context Growth Pattern

```
Call 1:  10 messages, 74KB
Call 5:  19 messages, 104KB
Call 10: 29 messages, 114KB
Call 15: 39 messages, 127KB
Call 21: 53 messages, 159KB

Growth: +2 messages per LLM call (assistant + tool result)
```

## Root Causes

### 1. No Context Pruning (`agent.py:177-208`)

The SDK accumulates ALL messages without truncation:
- Every tool call adds 2 messages (assistant + tool result)
- No history limits or summarization
- Only `max_turns=50` (turn limit, not context limit)

### 2. Worker Prompt File Loading (`prompts/worker.md:15-20`)

Each worker session reads all previous state:
- `eval_tasks.json` - Task list with status
- `state/analysis.json` - Initial trace analysis
- Generated code files - If task requires

**Note**: State files ARE necessary for worker orientation. Problem is MESSAGE HISTORY accumulation, not state files.

### 3. No SDK Session Resumption (`agent.py:369`)

Autonomous loop doesn't pass `session_id`:
```python
# Line 369: Calls query WITHOUT session_id
async for result in agent.query(prompt):  # session_id=None!
```

## Solution: Selective Message Pruning

### Strategy

Implement context window management that:
- **KEEPS**: System prompt + initial user prompt + state file contents + recent 3-5 turns
- **PRUNES**: Old assistant/tool message pairs beyond recent turns

### Implementation

#### 1. Context Pruning Function (`agent.py`)

```python
def _prune_context(self, messages: list, keep_recent_turns: int = 3) -> list:
    """
    Prune old messages while keeping critical context.

    Keep:
    - System prompt (index 0)
    - Initial user message (index 1)
    - Recent N conversation turns (last keep_recent_turns)
    """
    if len(messages) <= (2 + keep_recent_turns * 2):
        return messages  # Too short to prune

    keep_start = messages[:2]
    keep_end = messages[-(keep_recent_turns * 2):]

    pruned = keep_start + keep_end
    logger.info(f"Context pruned: {len(messages)} → {len(pruned)} messages")
    return pruned
```

**Challenge**: SDK manages messages internally. Need to research if SDK exposes message history for manipulation.

#### 2. Validation Retry Limit (`mlflow_ops.py`)

```python
MAX_TASK_ATTEMPTS = 5

def increment_task_attempts(task_id: int) -> bool:
    """
    Increment task attempt counter.
    Returns False if exceeded limit (fail task).
    """
    tasks_file = get_tasks_file()
    data = json.loads(tasks_file.read_text())

    for task in data["tasks"]:
        if task["id"] == task_id:
            attempts = task.get("attempts", 0) + 1
            task["attempts"] = attempts

            if attempts >= MAX_TASK_ATTEMPTS:
                task["status"] = "failed"
                task["error"] = f"Exceeded retry limit ({attempts} attempts)"
                tasks_file.write_text(json.dumps(data, indent=2))
                return False

            tasks_file.write_text(json.dumps(data, indent=2))
            return True
    return True
```

## Implementation Priority

### Phase 1: Quick Wins

1. **Add validation retry limit** (code-enforced, max 5 attempts)
   - Files: `src/mlflow_ops.py`

2. **Optimize worker prompt**
   - Add efficiency notes about when to read files
   - File: `prompts/worker.md`

### Phase 2: Context Pruning

3. **Investigate SDK context control**
   - Research `claude_agent_sdk` APIs for message pruning

4. **Implement context pruning function**
   - Add `_prune_context()` method
   - Files: `src/agent.py`

5. **Add context size monitoring**
   - Log message count and size per LLM call
   - File: `src/agent.py:230-250`

## Expected Impact

| Metric | Current | Target | Strategy |
|--------|---------|--------|----------|
| LLM calls (session_5) | 21 | <10 | Retry limit (5) + context efficiency |
| Context size per call | 117KB avg | <50KB | Message pruning (keep recent 3 turns) |
| Session duration | 236s | <120s | Fewer retries + smaller contexts |
| Cost per session | $1.29 | <$0.50 | Reduced token usage |

## Critical Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/agent.py` | 177-208 | ClaudeAgentOptions configuration |
| `src/agent.py` | 293-410 | Autonomous evaluation loop |
| `src/mlflow_ops.py` | 267-315 | File-based state management |
| `src/mlflow_ops.py` | 322-397 | Task tracking functions |
| `prompts/worker.md` | 15-20 | State file loading instructions |
| `src/config.py` | 35 | max_turns=50 setting |

## Verification

1. Run autonomous evaluation on same experiment
2. Compare metrics:
   - Session duration (target: <120s)
   - LLM calls per session (target: <10)
   - Context size per call (target: <50KB)
   - Total cost per session (target: <$0.50)
3. Check task completion - Ensure quality not degraded
4. Review traces - Verify context pruning working correctly

---

*Generated from trace analysis of `tr-c9426fa0a3e2a430688905a4ba8d66a1`*
