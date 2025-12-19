"""CLI entry point for MLflow Evaluation Agent."""

import argparse
import asyncio
import logging

from .agent import MLflowEvalAgent


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="MLflow Evaluation Agent")
    parser.add_argument("prompt", nargs="?", help="Prompt for the agent")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--analyze", help="Trace filter for analysis")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    agent = MLflowEvalAgent()

    if args.interactive:
        await run_interactive(agent)
    elif args.analyze:
        print(f"Analyzing traces: {args.analyze}")
        script = await agent.analyze_and_generate(filter_string=args.analyze)
        print(script)
    elif args.prompt:
        result = None
        async for result in agent.query(args.prompt):
            pass
        if result:
            print(result.response)
    else:
        parser.print_help()


async def run_interactive(agent: MLflowEvalAgent):
    """Run agent in interactive mode with session continuity."""
    print("MLflow Evaluation Agent (Interactive Mode)")
    print("Type 'quit' to exit, 'clear' to reset workspace, 'new' for new session")
    print("-" * 50)

    session_id = None

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if prompt.lower() == "quit":
            break
        if prompt.lower() == "clear":
            agent.clear_workspace()
            print("Workspace cleared.")
            continue
        if prompt.lower() == "new":
            session_id = None
            print("Starting new session.")
            continue
        if not prompt:
            continue

        print("\nAgent: ", end="", flush=True)
        result = None
        async for result in agent.query(prompt, session_id=session_id):
            pass

        if result:
            print(result.response)
            # Track session for follow-up queries
            if result.session_id:
                session_id = result.session_id
            if result.cost_usd:
                print(f"\n[Cost: ${result.cost_usd:.4f}]")
            if result.timing_metrics:
                print(f"[Timing: {result.timing_metrics.get('total_query_time', 0):.2f}s]")


def cli():
    """Synchronous entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
