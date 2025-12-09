---
description: Run agent tests with optional test file argument
allowed-tools: [Bash]
argument-hint: "[test-file-name]"
model: claude-haiku-4-5-20251001
---

Run agent tests and analyze results.

**Test file**: $ARGUMENTS (default: project-specific test file)

## Configuration

This command expects a test file in your project. Configure the default test file path by:
1. Providing the path as an argument: `/test-agent path/to/test.py`
2. Or setting up a standard location like `tests/test_agent.py`

## Instructions

1. **Determine test file**:
   - If argument provided: run the specified test file
   - If no argument: look for common test file patterns:
     - `tests/test_agent.py`
     - `test_agent.py`
     - `pytest tests/`

2. **Execute test**:
   ```bash
   # Run from project root
   python [test-file]
   # Or use pytest
   pytest [test-file] -v
   ```

3. **Extract key information**:
   - Test pass/fail status
   - Any error messages or stack traces
   - Trace IDs (look for "Trace ID:" in output)
   - Any warnings or recommendations

4. **Report results**:
   Provide a concise summary:
   - Status: PASS or FAIL
   - Trace IDs found (if any)
   - Key observations
   - Recommended next steps:
     - If PASS: "Use `/analyze-trace [trace-id]` to analyze trace"
     - If FAIL: Describe the issue and suggest fixes

5. **Follow-up actions** (do NOT execute automatically):
   - If tests passed, suggest running trace analysis
   - If tests failed, offer to fix the issue if user wants
