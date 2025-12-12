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

from src.agent import EvalAgentConfig, MLflowEvalAgent
from src.workspace import SharedWorkspace
from src.subagents import (
    AGENT_REGISTRY,
    TRACE_ANALYST_CONFIG,
    CONTEXT_ENGINEER_CONFIG,
    AGENT_ARCHITECT_CONFIG,
    create_subagents,
    validate_agent_can_run,
    get_coordinator_prompt,
    list_agents,
    get_workflow_order,
)


@dataclass
class TestCase:
    """A single test case with a prompt and expected behavior."""
    name: str
    prompt: str
    description: str
    expect_success: bool = True
    timeout_seconds: int = 120


# =============================================================================
# PHASE 1 & 2 TESTS - Registry & Selective Context (No MCP required)
# =============================================================================

def run_registry_tests() -> dict:
    """Run Phase 1 tests: Agent Registry.

    These tests verify that the registry pattern is working correctly.
    No MCP connection or Databricks credentials required.
    """
    print("\n" + "=" * 70)
    print("PHASE 1 TESTS: AGENT REGISTRY")
    print("=" * 70)

    results = {"passed": [], "failed": []}

    # Test 1: Registry is populated
    print("\n[1] Testing AGENT_REGISTRY is populated...")
    try:
        assert len(AGENT_REGISTRY) == 3, f"Expected 3 agents, got {len(AGENT_REGISTRY)}"
        assert "trace_analyst" in AGENT_REGISTRY
        assert "context_engineer" in AGENT_REGISTRY
        assert "agent_architect" in AGENT_REGISTRY
        results["passed"].append("registry_populated")
        print("    [PASS] Registry contains all 3 agents")
    except AssertionError as e:
        results["failed"].append({"name": "registry_populated", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 2: list_agents() works
    print("\n[2] Testing list_agents()...")
    try:
        agents = list_agents()
        assert len(agents) == 3
        assert "trace_analyst" in agents
        results["passed"].append("list_agents")
        print(f"    [PASS] list_agents() returns: {agents}")
    except AssertionError as e:
        results["failed"].append({"name": "list_agents", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 3: get_workflow_order() respects dependencies
    print("\n[3] Testing get_workflow_order() dependency ordering...")
    try:
        order = get_workflow_order()
        # trace_analyst has no dependencies, should be first
        assert order[0] == "trace_analyst", f"Expected trace_analyst first, got {order[0]}"
        # context_engineer depends on trace_analyst outputs
        trace_idx = order.index("trace_analyst")
        context_idx = order.index("context_engineer")
        assert context_idx > trace_idx, "context_engineer should come after trace_analyst"
        results["passed"].append("workflow_order")
        print(f"    [PASS] Workflow order: {' -> '.join(order)}")
    except (AssertionError, ValueError) as e:
        results["failed"].append({"name": "workflow_order", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 4: Agent configs have required fields
    print("\n[4] Testing agent configs have required fields...")
    try:
        for name, config in AGENT_REGISTRY.items():
            assert config.name == name, f"Config name mismatch: {config.name} != {name}"
            assert config.description, f"{name} missing description"
            assert config.prompt_template, f"{name} missing prompt_template"
            assert "{workspace_context}" in config.prompt_template, f"{name} prompt missing placeholder"
            assert config.tools, f"{name} has no tools"
        results["passed"].append("config_fields")
        print("    [PASS] All configs have required fields")
    except AssertionError as e:
        results["failed"].append({"name": "config_fields", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 5: Output keys are defined correctly
    print("\n[5] Testing output_keys definitions...")
    try:
        assert "trace_analysis_summary" in TRACE_ANALYST_CONFIG.output_keys
        assert "error_patterns" in TRACE_ANALYST_CONFIG.output_keys
        assert "context_recommendations" in CONTEXT_ENGINEER_CONFIG.output_keys
        # agent_architect tags traces directly, may have empty output_keys
        results["passed"].append("output_keys")
        print("    [PASS] Output keys correctly defined")
        print(f"           trace_analyst outputs: {TRACE_ANALYST_CONFIG.output_keys}")
        print(f"           context_engineer outputs: {CONTEXT_ENGINEER_CONFIG.output_keys}")
    except AssertionError as e:
        results["failed"].append({"name": "output_keys", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 6: Required keys define dependencies
    print("\n[6] Testing required_keys dependencies...")
    try:
        # trace_analyst has no dependencies (first in pipeline)
        assert TRACE_ANALYST_CONFIG.required_keys == [], \
            f"trace_analyst should have no required_keys, got {TRACE_ANALYST_CONFIG.required_keys}"
        # context_engineer depends on trace_analyst outputs
        assert "trace_analysis_summary" in CONTEXT_ENGINEER_CONFIG.required_keys
        assert "error_patterns" in CONTEXT_ENGINEER_CONFIG.required_keys
        results["passed"].append("required_keys")
        print("    [PASS] Dependencies correctly defined")
        print(f"           context_engineer requires: {CONTEXT_ENGINEER_CONFIG.required_keys}")
    except AssertionError as e:
        results["failed"].append({"name": "required_keys", "error": str(e)})
        print(f"    [FAIL] {e}")

    return results


def run_selective_context_tests() -> dict:
    """Run Phase 2 tests: Selective Context Injection.

    These tests verify that agents receive only relevant workspace data.
    No MCP connection or Databricks credentials required.
    """
    print("\n" + "=" * 70)
    print("PHASE 2 TESTS: SELECTIVE CONTEXT INJECTION")
    print("=" * 70)

    results = {"passed": [], "failed": []}

    # Test 1: validate_agent_can_run with empty workspace
    print("\n[1] Testing validate_agent_can_run() with empty workspace...")
    try:
        workspace = SharedWorkspace()

        # trace_analyst has no dependencies, should be able to run
        can_run, missing, msg = validate_agent_can_run("trace_analyst", workspace)
        assert can_run, f"trace_analyst should run with empty workspace: {msg}"

        # context_engineer has dependencies, should NOT be able to run
        can_run, missing, msg = validate_agent_can_run("context_engineer", workspace)
        assert not can_run, "context_engineer should NOT run with empty workspace"
        assert "trace_analysis_summary" in missing

        results["passed"].append("validate_empty_workspace")
        print("    [PASS] Dependency validation works for empty workspace")
    except AssertionError as e:
        results["failed"].append({"name": "validate_empty_workspace", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 2: validate_agent_can_run after dependencies populated
    print("\n[2] Testing validate_agent_can_run() with populated workspace...")
    try:
        workspace = SharedWorkspace()
        workspace.write("trace_analysis_summary", {"error_rate": 0.05}, agent="trace_analyst")
        workspace.write("error_patterns", [{"error_type": "timeout", "count": 1}], agent="trace_analyst")

        # Now context_engineer should be able to run
        can_run, missing, msg = validate_agent_can_run("context_engineer", workspace)
        assert can_run, f"context_engineer should run after dependencies met: {msg}"
        assert missing == []

        results["passed"].append("validate_populated_workspace")
        print("    [PASS] Dependency validation works after workspace populated")
    except AssertionError as e:
        results["failed"].append({"name": "validate_populated_workspace", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 3: Selective context includes only relevant keys
    print("\n[3] Testing selective context includes only relevant keys...")
    try:
        workspace = SharedWorkspace()
        workspace.write("trace_analysis_summary", {"error_rate": 0.05}, agent="trace_analyst")
        workspace.write("error_patterns", [{"error_type": "timeout", "count": 1}], agent="trace_analyst")
        workspace.write("context_recommendations", [{"issue": "prompt too long", "recommended_change": "shorten prompt"}], agent="context_engineer")
        workspace.write("unrelated_data", {"foo": "bar"}, agent="other")

        # context_engineer should get trace_analysis_summary and error_patterns
        # but NOT context_recommendations (its own output) or unrelated_data
        context = workspace.to_selective_context(CONTEXT_ENGINEER_CONFIG)

        assert "trace_analysis_summary" in context
        assert "error_patterns" in context
        # Own outputs should be excluded
        assert "context_recommendations" not in context or "other_available_keys" in context

        results["passed"].append("selective_context_filtering")
        print("    [PASS] Selective context filters correctly")
        print(f"           Context length: {len(context)} chars")
    except AssertionError as e:
        results["failed"].append({"name": "selective_context_filtering", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 4: create_subagents uses selective context
    print("\n[4] Testing create_subagents() uses selective context...")
    try:
        workspace = SharedWorkspace()
        workspace.write("trace_analysis_summary", {"error_rate": 0.05}, agent="trace_analyst")

        # Create with selective context (default)
        agents = create_subagents(workspace, use_selective_context=True)

        assert len(agents) == 3
        assert "trace_analyst" in agents
        assert "context_engineer" in agents
        assert "agent_architect" in agents

        # Verify each agent got a prompt
        for name, agent_def in agents.items():
            assert agent_def.prompt, f"{name} has no prompt"
            assert agent_def.description, f"{name} has no description"

        results["passed"].append("create_subagents_selective")
        print("    [PASS] create_subagents() works with selective context")
    except AssertionError as e:
        results["failed"].append({"name": "create_subagents_selective", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 5: Missing dependencies warning in context
    print("\n[5] Testing missing dependencies warning in context...")
    try:
        workspace = SharedWorkspace()
        # Don't populate required keys for context_engineer

        context = workspace.to_selective_context(CONTEXT_ENGINEER_CONFIG)

        assert "missing_dependencies" in context.lower() or "missing" in context.lower()

        results["passed"].append("missing_deps_warning")
        print("    [PASS] Missing dependencies warning appears in context")
    except AssertionError as e:
        results["failed"].append({"name": "missing_deps_warning", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 6: Token budget limits context size
    print("\n[6] Testing token budget limits context size...")
    try:
        workspace = SharedWorkspace()
        # Write a large entry
        large_data = {"items": [f"item_{i}" for i in range(1000)]}
        workspace.write("trace_analysis_summary", large_data, agent="trace_analyst")

        context = workspace.to_selective_context(CONTEXT_ENGINEER_CONFIG)

        # Should be truncated based on token budget
        max_expected = CONTEXT_ENGINEER_CONFIG.total_token_budget * 4 + 500  # Some overhead for XML
        assert len(context) < max_expected * 2, f"Context too large: {len(context)}"

        results["passed"].append("token_budget")
        print(f"    [PASS] Token budget respected (context: {len(context)} chars)")
    except AssertionError as e:
        results["failed"].append({"name": "token_budget", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 7: Coordinator prompt generated from registry
    print("\n[7] Testing coordinator prompt auto-generation...")
    try:
        workspace = SharedWorkspace()
        prompt = get_coordinator_prompt(workspace)

        # Should contain all agent names
        assert "trace_analyst" in prompt
        assert "context_engineer" in prompt
        assert "agent_architect" in prompt

        # Should contain workflow info
        assert "workflow" in prompt.lower() or "order" in prompt.lower()

        results["passed"].append("coordinator_prompt")
        print("    [PASS] Coordinator prompt auto-generated from registry")
        print(f"           Prompt length: {len(prompt)} chars")
    except AssertionError as e:
        results["failed"].append({"name": "coordinator_prompt", "error": str(e)})
        print(f"    [FAIL] {e}")

    # Test 8: Unknown agent handling
    print("\n[8] Testing unknown agent handling...")
    try:
        workspace = SharedWorkspace()
        can_run, missing, msg = validate_agent_can_run("nonexistent_agent", workspace)

        assert not can_run
        assert "Unknown agent" in msg

        results["passed"].append("unknown_agent")
        print("    [PASS] Unknown agent handled correctly")
    except AssertionError as e:
        results["failed"].append({"name": "unknown_agent", "error": str(e)})
        print(f"    [FAIL] {e}")

    return results


def run_phase_tests() -> dict:
    """Run all Phase 1 & 2 tests (no MCP required)."""
    print("\n" + "=" * 70)
    print("MLFLOW EVAL AGENT - PHASE 1 & 2 INTEGRATION TESTS")
    print("=" * 70)
    print("These tests verify the Agent Registry and Selective Context features.")
    print("No Databricks/MCP connection required.\n")

    registry_results = run_registry_tests()
    context_results = run_selective_context_tests()

    # Combine results
    all_passed = registry_results["passed"] + context_results["passed"]
    all_failed = registry_results["failed"] + context_results["failed"]

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 1 & 2 TEST SUMMARY")
    print("=" * 70)
    print(f"Phase 1 (Registry): {len(registry_results['passed'])} passed, {len(registry_results['failed'])} failed")
    print(f"Phase 2 (Context):  {len(context_results['passed'])} passed, {len(context_results['failed'])} failed")
    print("-" * 70)
    print(f"TOTAL: {len(all_passed)} passed, {len(all_failed)} failed")

    if all_failed:
        print("\nFailed tests:")
        for fail in all_failed:
            print(f"  - {fail['name']}: {fail['error'][:80]}")

    return {
        "passed": all_passed,
        "failed": all_failed,
        "registry": registry_results,
        "context": context_results,
    }


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
    python test_local_agent.py                       # Run all MCP integration tests
    python test_local_agent.py <test_name>           # Run specific test
    python test_local_agent.py --list                # List available tests
    python test_local_agent.py --prompt "..."        # Run custom prompt
    python test_local_agent.py -v --prompt "..."     # Verbose mode
    python test_local_agent.py --phase               # Run Phase 1 & 2 tests (no MCP required)
    python test_local_agent.py --phase1              # Run Phase 1 tests only (registry)
    python test_local_agent.py --phase2              # Run Phase 2 tests only (selective context)

Phase 1 & 2 tests (no Databricks credentials needed):
    --phase     Run all Phase 1 & 2 integration tests
    --phase1    Run Phase 1: Agent Registry tests
    --phase2    Run Phase 2: Selective Context Injection tests

MCP Integration tests (requires DATABRICKS_HOST, MLFLOW_EXPERIMENT_ID):
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

    elif args[0] == "--phase":
        # Run Phase 1 & 2 tests (no MCP required)
        results = run_phase_tests()
        sys.exit(0 if not results["failed"] else 1)

    elif args[0] == "--phase1":
        # Run Phase 1 tests only
        results = run_registry_tests()
        print(f"\nPhase 1: {len(results['passed'])} passed, {len(results['failed'])} failed")
        sys.exit(0 if not results["failed"] else 1)

    elif args[0] == "--phase2":
        # Run Phase 2 tests only
        results = run_selective_context_tests()
        print(f"\nPhase 2: {len(results['passed'])} passed, {len(results['failed'])} failed")
        sys.exit(0 if not results["failed"] else 1)

    elif args[0] == "--list":
        print("\nAvailable tests:")
        print("\nPhase 1 & 2 (no MCP required):")
        print("  --phase                     Run all Phase 1 & 2 tests")
        print("  --phase1                    Run Phase 1: Agent Registry")
        print("  --phase2                    Run Phase 2: Selective Context")
        print("\nMCP Integration tests:")
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
