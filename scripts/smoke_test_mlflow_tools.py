#!/usr/bin/env python
"""Smoke test for custom MLflow tools against real Databricks data.

Usage:
    uv run python scripts/smoke_test_mlflow_tools.py
"""

import asyncio
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.tools import create_mlflow_tools
from src.mlflow_client import clear_client_cache

# Use experiment ID from .env
EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID", "159502977489049")


def print_result(name: str, passed: bool, detail: str = ""):
    """Print test result with color."""
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"  [{status}] {name}")
    if detail and not passed:
        print(f"         {detail[:200]}")


async def main():
    print(f"\n{'='*60}")
    print("MLflow Tools Smoke Test")
    print(f"{'='*60}")
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Databricks Host: {os.getenv('DATABRICKS_HOST', 'not set')}")
    print(f"Config Profile: {os.getenv('DATABRICKS_CONFIG_PROFILE', 'not set')}")
    print(f"{'='*60}\n")

    # Clear any cached client
    clear_client_cache()

    # Create tools
    tools = create_mlflow_tools()
    print(f"Created {len(tools)} MLflow tools\n")

    # Get handlers
    search_traces = tools[0].handler
    get_trace = tools[1].handler
    set_trace_tag = tools[2].handler
    delete_trace_tag = tools[3].handler
    log_feedback = tools[4].handler
    log_expectation = tools[5].handler
    get_assessment = tools[6].handler
    update_assessment = tools[7].handler

    results = []
    trace_id = None

    # ==========================================================================
    # Test 1: search_traces
    # ==========================================================================
    print("1. Testing search_traces...")
    try:
        result = await search_traces({
            "experiment_id": EXPERIMENT_ID,
            "max_results": 5,
        })
        text = result["content"][0]["text"]
        passed = "trace" in text.lower() or "no traces" in text.lower()
        print_result("search_traces", passed, text)
        print(f"         Search results: {json.dumps(result, indent=2)}")
        results.append(("search_traces", passed))

        # Extract first trace_id for subsequent tests
        if "tr-" in text:
            # Parse from table format: | tr-xxx |
            for line in text.split("\n"):
                if "tr-" in line:
                    parts = line.split("|")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("tr-"):
                            trace_id = part
                            break
                    if trace_id:
                        break
        print(f"         Found trace_id: {trace_id}\n")

    except Exception as e:
        print_result("search_traces", False, str(e))
        results.append(("search_traces", False))

    # ==========================================================================
    # Test 2: get_trace
    # ==========================================================================
    print("2. Testing get_trace...")
    if trace_id:
        try:
            result = await get_trace({"trace_id": trace_id})
            text = result["content"][0]["text"]
            data = json.loads(text)
            passed = "info" in data and "spans" in data
            print_result("get_trace", passed, f"Got {len(data.get('spans', []))} spans")
            print(f"         Trace data: {json.dumps(data, indent=2)}")
            results.append(("get_trace", passed))

            # Show span types found
            span_types = set(s.get("span_type", "UNKNOWN") for s in data.get("spans", []))
            print(f"         Span types: {span_types}\n")

        except Exception as e:
            print_result("get_trace", False, str(e))
            results.append(("get_trace", False))
    else:
        print_result("get_trace", False, "No trace_id from search")
        results.append(("get_trace", False))

    # ==========================================================================
    # Test 3: set_trace_tag
    # ==========================================================================
    print("3. Testing set_trace_tag...")
    if trace_id:
        try:
            result = await set_trace_tag({
                "trace_id": trace_id,
                "key": "smoke_test",
                "value": "passed"
            })
            text = result["content"][0]["text"]
            passed = "tag" in text.lower() and "set" in text.lower()
            print_result("set_trace_tag", passed, text)
            print(f"         Set trace tag result: {json.dumps(result, indent=2)}")
            results.append(("set_trace_tag", passed))
        except Exception as e:
            print_result("set_trace_tag", False, str(e))
            results.append(("set_trace_tag", False))
    else:
        print_result("set_trace_tag", False, "No trace_id")
        results.append(("set_trace_tag", False))
    print()

    # ==========================================================================
    # Test 4: delete_trace_tag
    # ==========================================================================
    print("4. Testing delete_trace_tag...")
    if trace_id:
        try:
            result = await delete_trace_tag({
                "trace_id": trace_id,
                "key": "smoke_test"
            })
            text = result["content"][0]["text"]
            passed = "tag" in text.lower() and "deleted" in text.lower()
            print_result("delete_trace_tag", passed, text)
            print(f"         Delete trace tag result: {json.dumps(result, indent=2)}")
            results.append(("delete_trace_tag", passed))
        except Exception as e:
            print_result("delete_trace_tag", False, str(e))
            results.append(("delete_trace_tag", False))
    else:
        print_result("delete_trace_tag", False, "No trace_id")
        results.append(("delete_trace_tag", False))
    print()

    # ==========================================================================
    # Test 5: log_feedback
    # ==========================================================================
    print("5. Testing log_feedback...")
    if trace_id:
        try:
            result = await log_feedback({
                "trace_id": trace_id,
                "name": "smoke_test_feedback",
                "value": "test_value",
                "source_type": "CODE",
                "rationale": "Smoke test verification"
            })
            text = result["content"][0]["text"]
            passed = "feedback" in text.lower() and "logged" in text.lower()
            print_result("log_feedback", passed, text)
            print(f"         Log feedback result: {json.dumps(result, indent=2)}")
            results.append(("log_feedback", passed))
        except Exception as e:
            print_result("log_feedback", False, str(e))
            results.append(("log_feedback", False))
    else:
        print_result("log_feedback", False, "No trace_id")
        results.append(("log_feedback", False))
    print()

    # ==========================================================================
    # Test 6: get_assessment
    # ==========================================================================
    print("6. Testing get_assessment...")
    if trace_id:
        try:
            result = await get_assessment({
                "trace_id": trace_id,
                "assessment_name": "smoke_test_feedback"
            })
            text = result["content"][0]["text"]
            # Either found the assessment or reports not found (both valid responses)
            passed = "smoke_test_feedback" in text or "not found" in text.lower() or "no assessment" in text.lower()
            print_result("get_assessment", passed, text[:200])
            print(f"         Get assessment result: {json.dumps(result, indent=2)}")
            results.append(("get_assessment", passed))
        except Exception as e:
            print_result("get_assessment", False, str(e))
            results.append(("get_assessment", False))
    else:
        print_result("get_assessment", False, "No trace_id")
        results.append(("get_assessment", False))
    print()

    # ==========================================================================
    # Test 7: log_expectation
    # ==========================================================================
    print("7. Testing log_expectation...")
    if trace_id:
        try:
            result = await log_expectation({
                "trace_id": trace_id,
                "name": "smoke_test_expectation",
                "value": "expected_output_here"
            })
            text = result["content"][0]["text"]
            passed = "expectation" in text.lower() and "logged" in text.lower()
            print_result("log_expectation", passed, text)
            print(f"         Log expectation result: {json.dumps(result, indent=2)}")
            results.append(("log_expectation", passed))
        except Exception as e:
            print_result("log_expectation", False, str(e))
            results.append(("log_expectation", False))
    else:
        print_result("log_expectation", False, "No trace_id")
        results.append(("log_expectation", False))
    print()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"Passed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("\n\033[92mAll smoke tests passed!\033[0m")
        return 0
    else:
        print("\n\033[91mSome tests failed.\033[0m")
        for name, passed in results:
            if not passed:
                print(f"  - {name}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
