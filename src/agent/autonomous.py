"""Autonomous evaluation loop for the MLflow Evaluation Agent.

Implements the autonomous mode that builds complete evaluation suites
by iterating through initializer and worker sessions.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, AsyncIterator, Callable

logger = logging.getLogger(__name__)

AUTO_CONTINUE_DELAY_SECONDS = 3
MAX_INITIALIZER_ATTEMPTS = 3


@dataclass
class AutonomousEvent:
    """Event emitted during autonomous evaluation for UI streaming."""
    event_type: str  # "task_header" | "agent_result" | "progress" | "complete" | "error"
    iteration: Optional[int] = None
    task_name: Optional[str] = None
    phase: Optional[str] = None  # "initializer" | "worker"
    agent_result: Optional["AgentResult"] = None
    progress_data: Optional[dict] = None
    error_message: Optional[str] = None


async def stream_autonomous(
    experiment_id: str,
    max_iterations: Optional[int] = None,
    abort_check: Optional[Callable[[], bool]] = None,
) -> AsyncIterator[AutonomousEvent]:
    """Async generator that yields AutonomousEvent objects during evaluation.

    Args:
        experiment_id: MLflow experiment ID to analyze
        max_iterations: Maximum iterations (None = until complete)
        abort_check: Optional callback returning True to signal abort
    """
    from ..core.config import Config
    from ..core.runtime import detect_runtime, RuntimeContext, get_sessions_base_path
    from .agent import MLflowAgent
    from .prompts import load_prompt
    from .mlflow_ops import (
        all_tasks_complete,
        print_progress_summary,
        print_final_summary,
        set_session_dir,
        get_tasks_file,
        start_context_monitoring,
    )

    config = Config.from_env()
    config.experiment_id = experiment_id

    # Set up session directory (uses Volume in Databricks, local otherwise)
    sessions_base = get_sessions_base_path()
    session_dir = sessions_base / config.session_id
    set_session_dir(session_dir)

    logger.info(f"Session: {config.session_id}")
    logger.info(f"Output:  {session_dir}")

    # Check if first run (no task file exists in this session)
    is_first_run = not get_tasks_file().exists()
    initializer_attempts = 0

    if is_first_run:
        logger.info("=" * 60)
        logger.info("  INITIALIZER SESSION")
        logger.info("  Analyzing traces and creating task plan...")
        logger.info("=" * 60)

    iteration = 0
    while True:
        iteration += 1

        # Check abort signal
        if abort_check and abort_check():
            logger.info("Abort signaled. Stopping.")
            yield AutonomousEvent(event_type="complete", iteration=iteration)
            return

        # Check iteration limit
        if max_iterations and iteration > max_iterations:
            logger.info(f"Reached max iterations ({max_iterations}). Stopping.")
            yield AutonomousEvent(event_type="complete", iteration=iteration)
            return

        # Check if all tasks complete
        if not is_first_run and all_tasks_complete():
            print_final_summary()
            yield AutonomousEvent(event_type="complete", iteration=iteration)
            return

        phase = "initializer" if is_first_run else "worker"
        prompt_name = phase

        # Yield task header event
        yield AutonomousEvent(
            event_type="task_header",
            iteration=iteration,
            phase=phase,
            task_name=f"Session {iteration} ({prompt_name})",
        )

        import mlflow

        # Track each iteration as a sub-span
        with mlflow.start_span(name=f"session_{iteration}") as iter_span:
            iter_span.set_attribute("iteration", iteration)
            iter_span.set_attribute("phase", phase)

            # Fresh agent per session (Anthropic pattern)
            agent = MLflowAgent(config)

            # Choose prompt based on state
            prompt = load_prompt(prompt_name)
            prompt = prompt.replace("{experiment_id}", experiment_id)
            prompt = prompt.replace("{session_dir}", str(session_dir))

            # Start context monitoring for this session
            context_metrics = start_context_monitoring(
                session_id=f"{config.session_id}_iter{iteration}",
                initial_prompt=prompt
            )

            logger.info(f"--- Session {iteration} ({prompt_name}) ---")

            # Run session and stream output
            try:
                async for result in agent.query(prompt):
                    # Check abort between agent results
                    if abort_check and abort_check():
                        logger.info("Abort signaled during session.")
                        yield AutonomousEvent(event_type="complete", iteration=iteration)
                        return

                    yield AutonomousEvent(
                        event_type="agent_result",
                        iteration=iteration,
                        phase=phase,
                        agent_result=result,
                    )

                # Log final response
                if result and result.response:
                    logger.info(f"Response:\n{result.response}")
                    iter_span.set_attribute("response_length", len(result.response))

                    # Show cost if available
                    if result.cost_usd:
                        logger.info(f"[Cost: ${result.cost_usd:.4f}]")
                        iter_span.set_attribute("cost_usd", result.cost_usd)

                    # Propagate token tracking to session span
                    if result.usage_data:
                        usage = result.usage_data
                        iter_span.set_attribute("input_tokens", usage.get("input_tokens", 0))
                        iter_span.set_attribute("output_tokens", usage.get("output_tokens", 0))
                        iter_span.set_attribute("cache_creation_input_tokens", usage.get("cache_creation_input_tokens", 0))
                        iter_span.set_attribute("cache_read_input_tokens", usage.get("cache_read_input_tokens", 0))
                        iter_span.set_attribute("total_tokens",
                            usage.get("input_tokens", 0) + usage.get("output_tokens", 0))

                # Log context metrics to span
                if context_metrics:
                    iter_span.set_attribute("context_tool_calls", context_metrics.tool_calls)
                    iter_span.set_attribute("context_estimated_messages", context_metrics.estimated_messages)
                    iter_span.set_attribute("context_estimated_kb", context_metrics.estimated_context_kb)
                    logger.info(
                        f"[Context] {context_metrics.tool_calls} tool calls, "
                        f"~{context_metrics.estimated_messages} messages, "
                        f"~{context_metrics.estimated_context_kb:.1f}KB"
                    )

            except KeyboardInterrupt:
                iter_span.set_attribute("status", "interrupted")
                logger.info("Interrupted by user.")
                yield AutonomousEvent(event_type="complete", iteration=iteration)
                return
            except Exception as e:
                iter_span.set_attribute("error", str(e))
                logger.error(f"Error in session: {e}")
                logger.exception("Session error")
                yield AutonomousEvent(
                    event_type="error",
                    iteration=iteration,
                    error_message=str(e),
                )

            # After session completes, transition from initializer to worker
            if is_first_run:
                tasks_file = get_tasks_file()
                # Fallback: check if agent wrote to state/ directory via save_findings
                if not tasks_file.exists():
                    state_tasks = session_dir / "state" / "eval_tasks.json"
                    if state_tasks.exists():
                        import shutil
                        shutil.move(str(state_tasks), str(tasks_file))
                        logger.info(f"Moved task file from {state_tasks} to {tasks_file}")
                if tasks_file.exists():
                    is_first_run = False
                    logger.info("Initializer complete. Switching to worker mode.")
                else:
                    initializer_attempts += 1
                    if initializer_attempts >= MAX_INITIALIZER_ATTEMPTS:
                        logger.error(f"Initializer failed after {MAX_INITIALIZER_ATTEMPTS} attempts.")
                        yield AutonomousEvent(
                            event_type="error",
                            iteration=iteration,
                            error_message=f"Initializer failed to create task file after {MAX_INITIALIZER_ATTEMPTS} attempts.",
                        )
                        return
                    logger.warning(
                        f"Initializer did not create task file "
                        f"(attempt {initializer_attempts}/{MAX_INITIALIZER_ATTEMPTS})."
                    )

        # Progress summary
        print_progress_summary()

        # Yield progress event
        yield AutonomousEvent(
            event_type="progress",
            iteration=iteration,
            phase=phase,
            progress_data={
                "is_first_run": is_first_run,
                "session_id": config.session_id,
            },
        )

        # Auto-continue with interrupt window
        runtime = detect_runtime()

        if runtime.context == RuntimeContext.LOCAL:
            logger.info(f"Continuing in {AUTO_CONTINUE_DELAY_SECONDS}s (Ctrl+C to stop)...")
        else:
            logger.info(f"Continuing in {AUTO_CONTINUE_DELAY_SECONDS}s...")

        try:
            await asyncio.sleep(AUTO_CONTINUE_DELAY_SECONDS)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            yield AutonomousEvent(event_type="complete", iteration=iteration)
            return


async def run_autonomous(
    experiment_id: str,
    max_iterations: Optional[int] = None,
) -> None:
    """Run autonomous evaluation loop with task tracking.

    Args:
        experiment_id: MLflow experiment ID to analyze
        max_iterations: Maximum iterations (None = until complete)
    """
    import mlflow

    with mlflow.start_span(name="autonomous_evaluation", span_type="AGENT"):
        async for event in stream_autonomous(experiment_id, max_iterations):
            if event.event_type == "agent_result" and event.agent_result:
                result = event.agent_result
                if result.event_type == "text" and result.response:
                    pass  # Text accumulated
            elif event.event_type == "complete":
                break
