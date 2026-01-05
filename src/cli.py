"""Simplified CLI for MLflow Evaluation Agent.

Supports three modes:
- Interactive (-i): Free-form queries with session continuity
- Autonomous (-a): Auto-continue loop building complete eval suite
- Single query: One-shot prompt execution

~80 lines vs original ~90 lines
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

  # Single query
  python -m src.cli "Analyze traces in experiment 123"
        """
    )
    parser.add_argument("prompt", nargs="?", help="Prompt for the agent")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode with session continuity")
    parser.add_argument("--autonomous", "-a", action="store_true",
                        help="Autonomous mode with auto-continue loop")
    parser.add_argument("--experiment-id", "-e", type=str,
                        help="MLflow experiment ID (required for autonomous mode)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Max iterations for autonomous mode")
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
