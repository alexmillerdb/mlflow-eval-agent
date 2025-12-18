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

    args = parser.parse_args()

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
    """Synchronous CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
