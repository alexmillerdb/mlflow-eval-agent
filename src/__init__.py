"""MLflow Evaluation Agent package.

Simplified architecture following Anthropic best practices:
- Single agent with 3 tools (vs coordinator + 4 sub-agents + 11 tools)
- File-based state (vs 787-line workspace.py)
- External prompts (vs hardcoded)
"""

from .agent import MLflowAgent, AgentResult
from .config import Config

__all__ = [
    "MLflowAgent",
    "AgentResult",
    "Config",
]
