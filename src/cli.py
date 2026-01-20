"""Simplified CLI for MLflow Evaluation Agent.

Supports four modes:
- Interactive (-i): Free-form queries with session continuity
- Autonomous (-a): Auto-continue loop building complete eval suite
- Test (test <component>): Test components in isolation
- Single query: One-shot prompt execution
"""

import argparse
import asyncio
import logging
import os


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MLflow Evaluation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m src.cli -i

  # Autonomous mode (builds complete eval suite)
  python -m src.cli -a -e 123456789

  # Test components in isolation
  python -m src.cli test initializer -e 123456789
  python -m src.cli test worker -e 123456789 --task-type dataset
  python -m src.cli test tools mlflow_query --operation search
  python -m src.cli test integration -e 123456789 --max-iterations 2

  # Single query
  python -m src.cli "Analyze traces in experiment 123"
        """
    )

    # Create subparsers for test command
    subparsers = parser.add_subparsers(dest="command")

    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Test components in isolation")
    test_sub = test_parser.add_subparsers(dest="component")

    # test initializer
    init_p = test_sub.add_parser("initializer", help="Test initializer session")
    init_p.add_argument("-e", "--experiment-id", required=True, help="MLflow experiment ID")
    init_p.add_argument("--session-dir", type=str, help="Session directory (uses temp dir if not provided)")
    init_p.add_argument("--mock", action="store_true", help="Use mock MLflow client")
    init_p.add_argument("--tracking-uri", type=str, help="MLflow tracking URI")
    init_p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    init_p.add_argument("--background", action="store_true", help="Run in background, output to file")

    # test worker
    worker_p = test_sub.add_parser("worker", help="Test worker session")
    worker_p.add_argument("-e", "--experiment-id", required=True, help="MLflow experiment ID")
    worker_p.add_argument("--session-dir", type=str, help="Session directory with eval_tasks.json")
    worker_p.add_argument("--task-type", choices=["dataset", "scorer", "script", "validate", "fix"],
                          help="Filter to specific task type")
    worker_p.add_argument("--mock", action="store_true", help="Create mock tasks file if not exists")
    worker_p.add_argument("--tracking-uri", type=str, help="MLflow tracking URI")
    worker_p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    worker_p.add_argument("--background", action="store_true", help="Run in background, output to file")

    # test tools
    tools_p = test_sub.add_parser("tools", help="Test MCP tools directly")
    tools_p.add_argument("tool", choices=["mlflow_query", "mlflow_annotate", "save_findings"],
                         help="Tool to test")
    tools_p.add_argument("--operation", default="search",
                         help="Operation to perform (default: search)")
    tools_p.add_argument("-e", "--experiment-id", type=str, help="Experiment ID for search")
    tools_p.add_argument("--trace-id", type=str, help="Trace ID for get/annotate")
    tools_p.add_argument("--mock", action="store_true", help="Use mock MLflow client")
    tools_p.add_argument("--tracking-uri", type=str, help="MLflow tracking URI")
    tools_p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # test integration
    int_p = test_sub.add_parser("integration", help="Run full integration test")
    int_p.add_argument("-e", "--experiment-id", required=True, help="MLflow experiment ID")
    int_p.add_argument("--max-iterations", type=int, default=2,
                       help="Max iterations (default: 2)")
    int_p.add_argument("--mock", action="store_true", help="Use mock MLflow client")
    int_p.add_argument("--tracking-uri", type=str, help="MLflow tracking URI")
    int_p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    int_p.add_argument("--background", action="store_true", help="Run in background, output to file")

    # Main parser arguments (for non-test modes)
    parser.add_argument("prompt", nargs="?", help="Prompt for the agent")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode with session continuity")
    parser.add_argument("--autonomous", "-a", action="store_true",
                        help="Autonomous mode with auto-continue loop")
    parser.add_argument("--experiment-id", "-e", type=str,
                        help="MLflow experiment ID (required for autonomous mode)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Max iterations for autonomous mode")
    parser.add_argument("--tracking-uri", type=str,
                        help="MLflow tracking URI (default: databricks)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    # Databricks job parameters (mapped to env vars for config.py/runtime.py)
    parser.add_argument("--volume-path", type=str,
                        help="Unity Catalog Volume path for session storage")
    parser.add_argument("--anthropic-base-url", type=str,
                        help="Anthropic API base URL")
    parser.add_argument("--anthropic-auth-token", type=str,
                        help="Anthropic auth token")
    parser.add_argument("--anthropic-api-key", type=str,
                        help="Anthropic API key")
    parser.add_argument("--model", type=str,
                        help="Model to use")
    parser.add_argument("--mlflow-agent-experiment-id", type=str,
                        help="MLflow experiment ID for agent traces")
    parser.add_argument("--claude-code-disable-experimental-betas", type=str,
                        help="Disable Claude Code experimental betas")
    parser.add_argument("--secret-scope", type=str,
                        help="Databricks secret scope for auth token")
    parser.add_argument("--secret-key", type=str,
                        help="Databricks secret key for auth token")

    args = parser.parse_args()

    # Handle test command
    if args.command == "test":
        await handle_test_command(args)
        return

    # Handle tracking URI for non-test modes
    if hasattr(args, "tracking_uri") and args.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri

    # Read auth token from Databricks secret if scope/key provided
    if args.secret_scope and args.secret_key:
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            token = w.dbutils.secrets.get(scope=args.secret_scope, key=args.secret_key)
            os.environ["ANTHROPIC_AUTH_TOKEN"] = token
            logging.info(f"Loaded auth token from secret {args.secret_scope}/{args.secret_key}")
        except Exception as e:
            logging.error(f"Failed to read secret {args.secret_scope}/{args.secret_key}: {e}")

    # Map CLI args to environment variables (for config.py/runtime.py)
    env_mappings = [
        ("volume_path", "MLFLOW_AGENT_VOLUME_PATH"),
        ("anthropic_base_url", "ANTHROPIC_BASE_URL"),
        ("anthropic_auth_token", "ANTHROPIC_AUTH_TOKEN"),
        ("anthropic_api_key", "ANTHROPIC_API_KEY"),
        ("model", "DABS_MODEL"),
        ("mlflow_agent_experiment_id", "MLFLOW_AGENT_EXPERIMENT_ID"),
        ("claude_code_disable_experimental_betas", "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"),
    ]
    for arg_name, env_var in env_mappings:
        value = getattr(args, arg_name, None)
        if value is not None:
            os.environ[env_var] = value

    # Initialize MLflow AFTER env vars are set (critical for experiment tracing)
    from .agent import setup_mlflow
    setup_mlflow()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.autonomous:
        # Autonomous mode - builds complete eval suite
        experiment_id = args.experiment_id or os.getenv("MLFLOW_EXPERIMENT_ID")
        if not experiment_id:
            print("Error: --experiment-id or MLFLOW_EXPERIMENT_ID required for autonomous mode")
            return

        from .agent import run_autonomous
        await run_autonomous(experiment_id, args.max_iterations)

    elif args.interactive:
        # Interactive mode - free-form queries
        # Guard against running in non-interactive environments (Databricks Jobs)
        from .runtime import detect_runtime, RuntimeContext

        runtime = detect_runtime()
        if runtime.context == RuntimeContext.DATABRICKS_JOB:
            logging.error("Interactive mode not supported in Databricks Jobs")
            logging.error("Use --autonomous mode instead: mlflow-eval -a -e <experiment_id>")
            return

        from .agent import MLflowAgent
        agent = MLflowAgent()
        await run_interactive(agent)

    elif args.prompt:
        # Single query mode
        from .agent import MLflowAgent
        agent = MLflowAgent()
        async for result in agent.query(args.prompt):
            pass
        if result:
            print(result.response)
    else:
        parser.print_help()


async def run_interactive(agent):
    """Interactive mode with session continuity."""
    print("MLflow Evaluation Agent (Interactive)")
    print("Commands: 'quit', 'clear', 'new'")
    print("-" * 40)

    session_id = None

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() == "quit":
            break
        if prompt.lower() == "clear":
            agent.clear_state()
            print("State cleared.")
            continue
        if prompt.lower() == "new":
            session_id = None
            print("New session started.")
            continue
        if not prompt:
            continue

        print("\nAgent: ", end="", flush=True)
        result = None
        async for result in agent.query(prompt, session_id=session_id):
            pass

        if result:
            print(result.response)
            if result.session_id:
                session_id = result.session_id
            if result.cost_usd:
                print(f"\n[Cost: ${result.cost_usd:.4f}]")
            if result.duration_ms:
                print(f"[Duration: {result.duration_ms}ms]")


def _make_progress_callback():
    """Create a progress callback for streaming test output."""
    start_time = None

    def print_progress(event_type: str, data: dict):
        nonlocal start_time
        import time

        if start_time is None:
            start_time = time.time()

        if event_type == "tool_use":
            tool_name = data.get("tool_name", "unknown")
            if tool_name:
                print(f"  \u2192 {tool_name}")
        elif event_type == "result":
            elapsed = int(time.time() - start_time) if start_time else 0
            print(f"  \u2713 Complete ({elapsed}s)")

    return print_progress


def _run_in_background(component: str, args) -> None:
    """Launch test in background subprocess."""
    import subprocess
    import sys
    import tempfile
    from uuid import uuid4
    from pathlib import Path

    output_file = Path(tempfile.gettempdir()) / f"mlflow-test-{uuid4().hex[:8]}.log"

    # Build command to run test in foreground mode (no --background)
    cmd = [sys.executable, "-m", "src.cli", "test", component]
    cmd.extend(["-e", args.experiment_id])

    if hasattr(args, "session_dir") and args.session_dir:
        cmd.extend(["--session-dir", args.session_dir])
    if hasattr(args, "mock") and args.mock:
        cmd.append("--mock")
    if hasattr(args, "tracking_uri") and args.tracking_uri:
        cmd.extend(["--tracking-uri", args.tracking_uri])
    if hasattr(args, "verbose") and args.verbose:
        cmd.append("-v")
    if hasattr(args, "task_type") and args.task_type:
        cmd.extend(["--task-type", args.task_type])
    if hasattr(args, "max_iterations") and args.max_iterations:
        cmd.extend(["--max-iterations", str(args.max_iterations)])

    # Run in background
    with open(output_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    print(f"Running in background (PID: {proc.pid})")
    print(f"Output: {output_file}")
    print(f"Check status: tail -f {output_file}")


async def handle_test_command(args):
    """Handle test subcommand for component testing."""
    from pathlib import Path

    from .test_harness import (
        run_initializer_session,
        run_worker_session,
        test_tool_direct,
        run_integration_test,
        create_mock_tasks,
        create_mock_analysis,
        print_test_result,
    )

    # Set tracking URI if provided
    if hasattr(args, "tracking_uri") and args.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri

    # Set verbose logging if requested
    verbose = getattr(args, "verbose", False)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    component = args.component
    background = getattr(args, "background", False)

    # Handle background mode for supported components
    if background and component in ("initializer", "worker", "integration"):
        _run_in_background(component, args)
        return

    # Create progress callback for streaming output
    on_progress = _make_progress_callback() if not verbose else None

    if component == "initializer":
        print(f"Testing initializer session for experiment {args.experiment_id}...")
        session_dir = Path(args.session_dir) if args.session_dir else None
        result = await run_initializer_session(
            experiment_id=args.experiment_id,
            session_dir=session_dir,
            mock=args.mock,
            on_progress=on_progress,
        )
        print_test_result(result, verbose=verbose)

    elif component == "worker":
        print(f"Testing worker session for experiment {args.experiment_id}...")
        session_dir = Path(args.session_dir) if args.session_dir else None

        # Create mock tasks if requested and session_dir is provided
        if args.mock and session_dir:
            from .mlflow_ops import get_tasks_file, set_session_dir
            set_session_dir(session_dir)
            if not get_tasks_file().exists():
                print("Creating mock tasks file...")
                create_mock_tasks(session_dir)
                create_mock_analysis(session_dir, args.experiment_id)

        if session_dir is None:
            print("Error: --session-dir required for worker test (or use --mock to create temp session)")
            return

        result = await run_worker_session(
            experiment_id=args.experiment_id,
            session_dir=session_dir,
            task_type=args.task_type,
            mock=args.mock,
            on_progress=on_progress,
        )
        print_test_result(result, verbose=verbose)

    elif component == "tools":
        print(f"Testing tool {args.tool} with operation {args.operation}...")
        result = await test_tool_direct(
            tool_name=args.tool,
            operation=args.operation,
            experiment_id=args.experiment_id,
            trace_id=args.trace_id,
            mock=args.mock,
        )
        print_test_result(result, verbose=verbose)

    elif component == "integration":
        print(f"Running integration test for experiment {args.experiment_id}...")
        print(f"Max iterations: {args.max_iterations}")
        result = await run_integration_test(
            experiment_id=args.experiment_id,
            max_iterations=args.max_iterations,
            mock=args.mock,
            on_progress=on_progress,
        )
        print_test_result(result, verbose=verbose)

    else:
        print(f"Unknown test component: {component}")
        print("Available components: initializer, worker, tools, integration")


def cli():
    """Synchronous CLI entry point - works in CLI and Databricks Jobs."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop - standard CLI usage
        asyncio.run(main())
    else:
        # Event loop already running (Databricks/IPython kernel)
        # Use nest_asyncio to allow nested event loops
        import nest_asyncio
        nest_asyncio.apply()
        loop.run_until_complete(main())


if __name__ == "__main__":
    cli()
