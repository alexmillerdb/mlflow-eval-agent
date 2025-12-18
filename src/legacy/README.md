# Legacy Code - Deprecated 2025-12-18

This directory contains the original coordinator + sub-agent architecture.
It was deprecated in favor of a simplified single-agent approach.

## What's Here

| File/Directory | Description | Lines |
|---------------|-------------|-------|
| `workspace.py` | Inter-agent communication with Pydantic validation, TTL caching | 787 |
| `tool_compression.py` | Multi-mode output truncation (PREVIEW, SUMMARY, AGGRESSIVE, FULL) | 398 |
| `mlflow_client.py` | Adaptive span-type truncation | 263 |
| `subagents/` | Coordinator + 4 specialized sub-agents + registry | 1,292 |

**Total: ~2,740 lines**

## Why Deprecated

This architecture was over-engineered for MVP needs and did not follow best practices:

1. **Tool Overload**: 11 tools caused tool selection errors
2. **Workspace Complexity**: Pydantic validation + TTL caching for simple state
3. **Multiple Truncation Modes**: 4 modes across 3 files instead of single strategy
4. **Coordinator Pattern**: Complex delegation for 4 sub-agents

## Best Practices Applied

See:
- [Anthropic: Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Production-Grade Agentic AI Workflows](https://arxiv.org/html/2512.08769v1)

Key principles:
- Single agent with good prompts
- One-agent-one-tool (or minimal tools)
- File-based state vs in-memory complexity
- Session isolation with fresh context

## To Restore

```bash
git checkout v0.1-coordinator-architecture
```

Or manually copy files back from this directory.
