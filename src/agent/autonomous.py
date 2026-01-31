"""Autonomous evaluation loop for the MLflow Evaluation Agent.

Implements the autonomous mode that builds complete evaluation suites
by iterating through initializer and worker sessions.
"""

import asyncio
import logging
from typing import Optional

import mlflow

logger = logging.getLogger(__name__)

AUTO_CONTINUE_DELAY_SECONDS = 3


@mlflow.trace(name="autonomous_evaluation", span_type="AGENT")
async def run_autonomous(
    experiment_id: str,
    max_iterations: Optional[int] = None,
) -> None:
    """Run autonomous evaluation loop with task tracking.

    Args:
        experiment_id: MLflow experiment ID to analyze
        max_iterations: Maximum iterations (None = until complete)
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

    if is_first_run:
        logger.info("=" * 60)
        logger.info("  INITIALIZER SESSION")
        logger.info("  Analyzing traces and creating task plan...")
        logger.info("=" * 60)

    iteration = 0
    while True:
        iteration += 1

        # Check iteration limit
        if max_iterations and iteration > max_iterations:
            logger.info(f"Reached max iterations ({max_iterations}). Stopping.")
            break

        # Check if all tasks complete
        if not is_first_run and all_tasks_complete():
            print_final_summary()
            break

        # Track each iteration as a sub-span
        with mlflow.start_span(name=f"session_{iteration}") as iter_span:
            iter_span.set_attribute("iteration", iteration)
            iter_span.set_attribute("phase", "initializer" if is_first_run else "worker")

            # Fresh agent per session (Anthropic pattern)
            agent = MLflowAgent(config)

            # Choose prompt based on state
            prompt_name = "initializer" if is_first_run else "worker"
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
                    if result.event_type == "text":
                        # Print incremental text (clear line and reprint for streaming effect)
                        pass  # Text is accumulated in result.response

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
                break
            except Exception as e:
                iter_span.set_attribute("error", str(e))
                logger.error(f"Error in session: {e}")
                logger.exception("Session error")

            # After session completes, transition from initializer to worker
            if is_first_run:
                tasks_file = get_tasks_file()
                if tasks_file.exists():
                    is_first_run = False
                    logger.info("✓ Initializer session complete. Switching to worker mode.")
                else:
                    logger.warning("⚠ Initializer did not create task file. Will retry initializer session.")

        # Progress summary
        print_progress_summary()

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
            break
