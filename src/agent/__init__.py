"""Agent implementation for MLflow Evaluation Agent.

Contains the main agent class, tools, and MLflow operations.
"""

from .agent import MLflowAgent, AgentResult, setup_mlflow
from .autonomous import run_autonomous
from .prompts import load_prompt
from .tools import create_tools, MCPTools, BuiltinTools
from .mlflow_ops import (
    ContextMetrics,
    start_context_monitoring,
    get_context_metrics,
    record_tool_call,
    _reset_context_metrics,
    search_traces,
    get_trace,
    search_runs,
    get_run,
    set_tag,
    log_feedback,
    log_expectation,
    save_state,
    load_state,
    clear_state,
    set_session_dir,
    get_session_dir,
    get_tasks_file,
    get_state_dir,
    get_evaluation_dir,
    get_task_status,
    all_tasks_complete,
    print_progress_summary,
    print_final_summary,
)

__all__ = [
    # Agent
    "MLflowAgent",
    "AgentResult",
    "setup_mlflow",
    "run_autonomous",
    "load_prompt",
    # Tools
    "create_tools",
    "MCPTools",
    "BuiltinTools",
    # Context monitoring
    "ContextMetrics",
    "start_context_monitoring",
    "get_context_metrics",
    "record_tool_call",
    "_reset_context_metrics",
    # MLflow operations
    "search_traces",
    "get_trace",
    "search_runs",
    "get_run",
    "set_tag",
    "log_feedback",
    "log_expectation",
    # State management
    "save_state",
    "load_state",
    "clear_state",
    "set_session_dir",
    "get_session_dir",
    "get_tasks_file",
    "get_state_dir",
    "get_evaluation_dir",
    # Task tracking
    "get_task_status",
    "all_tasks_complete",
    "print_progress_summary",
    "print_final_summary",
]
