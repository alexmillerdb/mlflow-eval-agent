"""
Integration test for MLflow Evaluation Agent.

Tests different prompts against the local agent.
Run with: uv run python src/test_local_agent.py

Environment variables required:
    DATABRICKS_HOST - Databricks workspace URL
    DATABRICKS_TOKEN - Databricks access token (optional if using profile)
    MLFLOW_EXPERIMENT_ID - MLflow experiment ID
"""

import asyncio
import sys
import time
from dataclasses import dataclass

from agent import EvalAgentConfig, MLflowEvalAgent


@dataclass
class TestCase:
    """A single test case with a prompt and expected behavior."""
    name: str
    prompt: str
    description: str
    expect_success: bool = True
    timeout_seconds: int = 120


# =============================================================================
# TEST CASES - Different prompt categories
# =============================================================================

TEST_CASES = [
    # ----- Basic Connectivity -----
    TestCase(
        name="list_tools",
        prompt="List the MCP tools you have access to. Be brief.",
        description="Verify basic agent connectivity and tool access",
    ),

    # ----- Trace Analysis Prompts -----
    TestCase(
        name="trace_search_recent",
        prompt="Search for the 5 most recent traces in the experiment. Summarize briefly.",
        description="Test basic trace search functionality",
    ),
    TestCase(
        name="trace_search_errors",
        prompt="Search for any ERROR traces from the last 24 hours. If none found, say so.",
        description="Test filtering traces by status",
    ),
    TestCase(
        name="trace_latency_analysis",
        prompt="Find the slowest traces (by execution time) and identify which components are bottlenecks.",
        description="Test performance analysis capabilities",
    ),

    # ----- Sub-Agent Invocation -----
    TestCase(
        name="invoke_trace_analyst",
        prompt="""
        Use the trace_analyst sub-agent to:
        1. Search for recent traces
        2. Write a summary to the shared workspace with key 'trace_analysis_summary'
        Be concise.
        """,
        description="Test trace_analyst sub-agent invocation",
    ),
    TestCase(
        name="invoke_context_engineer",
        prompt="""
        Use the context_engineer sub-agent to:
        1. Read any findings from the workspace
        2. Suggest one context optimization
        Be brief.
        """,
        description="Test context_engineer sub-agent invocation",
    ),

    # ----- Evaluation Script Generation -----
    TestCase(
        name="generate_simple_eval",
        prompt="""
        Generate a simple MLflow evaluation script that:
        1. Creates a dataset with 2 test cases
        2. Uses the Correctness scorer
        3. Prints the results

        Return only the Python code, no explanation.
        """,
        description="Test evaluation script generation",
    ),

    # ----- Workspace Communication -----
    TestCase(
        name="workspace_write_read",
        prompt="""
        1. Write test data to the workspace with key 'test_entry'
        2. Read it back and confirm it worked
        Be brief.
        """,
        description="Test workspace read/write functionality",
    ),

    # ----- Complex Analysis -----
    TestCase(
        name="full_analysis_workflow",
        prompt="""
        Perform a quick analysis:
        1. Use trace_analyst to check for any issues in recent traces
        2. Summarize key findings in 2-3 bullet points
        """,
        description="Test multi-step analysis workflow",
        timeout_seconds=180,
    ),
]


# =============================================================================
# TEST RUNNER
# =============================================================================

async def run_test(agent: MLflowEvalAgent, test: TestCase) -> tuple[bool, str, float]:
    """
    Run a single test case.

    Returns: (passed, response_or_error, duration_seconds)
    """
    print(f"\n{'='*60}")
    print(f"TEST: {test.name}")
    print(f"{'='*60}")
    print(f"Description: {test.description}")
    print(f"Prompt: {test.prompt[:100]}...")
    print("-" * 60)

    start_time = time.time()

    try:
        final_result = None
        async for result in agent.query(test.prompt):
            final_result = result
            # Print progress dots
            print(".", end="", flush=True)

        duration = time.time() - start_time
        print()  # Newline after dots

        if final_result is None:
            return False, "No response received", duration

        if final_result.success != test.expect_success:
            return False, f"Expected success={test.expect_success}, got {final_result.success}", duration

        # Print response summary
        response_preview = final_result.response[:500] if final_result.response else "(empty)"
        print(f"\nResponse preview:\n{response_preview}")

        if final_result.cost_usd:
            print(f"\nCost: ${final_result.cost_usd:.4f}")
        if final_result.timing_metrics:
            print(f"Duration: {final_result.timing_metrics.get('total_query_time', duration):.2f}s")

        return True, final_result.response, duration

    except asyncio.TimeoutError:
        duration = time.time() - start_time
        return False, f"Timeout after {test.timeout_seconds}s", duration
    except Exception as e:
        duration = time.time() - start_time
        return False, f"Exception: {type(e).__name__}: {e}", duration


async def run_all_tests(test_names: list[str] | None = None) -> dict:
    """
    Run all tests or a subset by name.

    Args:
        test_names: Optional list of test names to run. If None, runs all.

    Returns:
        Summary dict with results.
    """
    print("\n" + "=" * 70)
    print("MLFLOW EVALUATION AGENT - INTEGRATION TESTS")
    print("=" * 70)

    # Initialize agent
    try:
        config = EvalAgentConfig.from_env()
        print(f"\nDatabricks Host: {config.databricks_host}")
        print(f"Experiment ID: {config.experiment_id}")
        print(f"Model: {config.model}")
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("Please set required environment variables.")
        return {"error": str(e)}

    agent = MLflowEvalAgent(config=config)

    # Filter tests if specified
    tests_to_run = TEST_CASES
    if test_names:
        tests_to_run = [t for t in TEST_CASES if t.name in test_names]
        if not tests_to_run:
            print(f"\nNo tests found matching: {test_names}")
            print(f"Available tests: {[t.name for t in TEST_CASES]}")
            return {"error": "No matching tests"}

    print(f"\nRunning {len(tests_to_run)} test(s)...")

    # Run tests
    results = {
        "passed": [],
        "failed": [],
        "total_duration": 0,
        "total_cost": 0,
    }

    for test in tests_to_run:
        passed, response, duration = await asyncio.wait_for(
            run_test(agent, test),
            timeout=test.timeout_seconds,
        )

        results["total_duration"] += duration

        if passed:
            results["passed"].append({
                "name": test.name,
                "duration": duration,
            })
            print(f"\n[PASS] {test.name} ({duration:.1f}s)")
        else:
            results["failed"].append({
                "name": test.name,
                "error": response,
                "duration": duration,
            })
            print(f"\n[FAIL] {test.name} ({duration:.1f}s)")
            print(f"       Error: {response[:200]}")

        # Clear workspace between tests to avoid interference
        agent.clear_workspace()

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {len(results['passed'])}/{len(tests_to_run)}")
    print(f"Failed: {len(results['failed'])}/{len(tests_to_run)}")
    print(f"Total Duration: {results['total_duration']:.1f}s")

    if results["failed"]:
        print("\nFailed tests:")
        for fail in results["failed"]:
            print(f"  - {fail['name']}: {fail['error'][:100]}")

    return results


async def heartbeat(start_time: float, stop_event: asyncio.Event) -> None:
    """Print heartbeat every few seconds to show the agent is still working."""
    interval = 5  # seconds
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        if not stop_event.is_set():
            elapsed = time.time() - start_time
            print(f" [{elapsed:.0f}s]", end="", flush=True)


async def run_single_prompt(prompt: str, verbose: bool = False) -> None:
    """Run a single custom prompt for testing."""
    print("\n" + "=" * 70)
    print("SINGLE PROMPT TEST")
    print("=" * 70)

    try:
        config = EvalAgentConfig.from_env()
        print(f"Host: {config.databricks_host}")
        print(f"Experiment: {config.experiment_id}")
        print(f"Model: {config.model}")
        agent = MLflowEvalAgent(config=config)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return

    print(f"\nPrompt: {prompt}")
    print("-" * 70)
    print("Working", end="", flush=True)

    start_time = time.time()
    final_result = None
    chunk_count = 0

    # Start heartbeat task to show progress
    stop_heartbeat = asyncio.Event()
    heartbeat_task = asyncio.create_task(heartbeat(start_time, stop_heartbeat))

    try:
        async for result in agent.query(prompt):
            final_result = result
            chunk_count += 1

            # Handle different event types
            if result.event_type == "text":
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"\n[{elapsed:.1f}s] Text: {len(result.response)} chars", flush=True)
                else:
                    print(".", end="", flush=True)

            elif result.event_type == "tool_use":
                print(f"\n  -> Tool: {result.tool_name}", flush=True)
                if verbose and result.tool_input:
                    # Show truncated input for verbose mode
                    input_str = str(result.tool_input)
                    if len(input_str) > 100:
                        input_str = input_str[:100] + "..."
                    print(f"     Input: {input_str}", flush=True)

            elif result.event_type == "tool_result":
                status = "error" if result.tool_is_error else "ok"
                if result.tool_result:
                    preview = result.tool_result[:100] + "..." if len(result.tool_result) > 100 else result.tool_result
                else:
                    preview = "(empty)"
                print(f"  <- Result ({status}): {preview}", flush=True)

            elif result.event_type == "todo_update":
                print("\n  Todo List:", flush=True)
                for todo in result.todos or []:
                    status_icons = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}
                    icon = status_icons.get(todo.get("status"), "[ ]")
                    text = todo.get("activeForm") if todo.get("status") == "in_progress" else todo.get("content")
                    print(f"    {icon} {text}", flush=True)

            elif result.event_type == "subagent":
                print(f"\n  >> Subagent: {result.subagent_name}", flush=True)

            elif result.event_type == "result":
                # Final result - will be handled below
                pass

    finally:
        stop_heartbeat.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    duration = time.time() - start_time
    print(f"\n\nCompleted in {duration:.1f}s ({chunk_count} events)")
    print("-" * 70)

    if final_result:
        print(f"\nResponse:\n{final_result.response}")
        print(f"\nSuccess: {final_result.success}")
        if final_result.cost_usd:
            print(f"Cost: ${final_result.cost_usd:.4f}")
        if final_result.timing_metrics:
            print(f"Timing: {final_result.timing_metrics}")
    else:
        print("No response received")


# =============================================================================
# MAIN
# =============================================================================

def print_usage():
    """Print usage information."""
    print("""
Usage:
    python test_local_agent.py                       # Run all tests
    python test_local_agent.py <test_name>           # Run specific test
    python test_local_agent.py --list                # List available tests
    python test_local_agent.py --prompt "..."        # Run custom prompt
    python test_local_agent.py -v --prompt "..."     # Verbose mode

Available tests:
""")
    for test in TEST_CASES:
        print(f"    {test.name:25} - {test.description}")


def main():
    args = sys.argv[1:]
    verbose = "-v" in args or "--verbose" in args
    args = [a for a in args if a not in ("-v", "--verbose")]

    if not args:
        # Run all tests
        asyncio.run(run_all_tests())

    elif args[0] == "--list":
        print("\nAvailable tests:")
        for test in TEST_CASES:
            print(f"  {test.name:25} - {test.description}")

    elif args[0] == "--prompt":
        if len(args) < 2:
            print("Error: --prompt requires a prompt string")
            sys.exit(1)
        prompt = " ".join(args[1:])
        asyncio.run(run_single_prompt(prompt, verbose=verbose))

    elif args[0] == "--help" or args[0] == "-h":
        print_usage()

    else:
        # Run specific test(s)
        asyncio.run(run_all_tests(test_names=args))


if __name__ == "__main__":
    main()
