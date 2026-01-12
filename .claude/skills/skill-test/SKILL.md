---
name: skill-test
description: |
  Interactive skill testing and ground truth generation. Use when testing a skill's
  code generation capabilities by executing generated code and saving verified working
  examples. Invoke with: /skill-test <skill-name>
---

# /skill-test - Interactive Skill Testing

Test skills by executing generated code and building verified ground truth datasets.

## Usage

```
/skill-test <skill-name>
```

## Workflow

### 1. Load Target Skill

Read the target skill's SKILL.md and key references:

```
.claude/skills/<skill-name>/SKILL.md
.claude/skills/<skill-name>/references/  (if exists)
```

### 2. Get Test Prompt

Ask the user for a test prompt. Example prompts for mlflow-evaluation:
- "Generate an evaluation script for a RAG agent"
- "Create a custom scorer that checks code syntax"
- "Build a dataset from production traces"

### 3. Generate Response

Respond using the loaded skill's knowledge. Include complete, runnable Python code.

### 4. Execute Code Blocks

Extract Python code blocks from the response. For each block:

1. Write to temp file:
   ```bash
   cat > /tmp/skill_test_block_N.py << 'EOF'
   <code>
   EOF
   ```

2. Execute with timeout:
   ```bash
   timeout 30 python /tmp/skill_test_block_N.py
   ```

3. Report result:
   - Success: "Block N: Executed successfully"
   - Failure: "Block N: Failed - <error>"

### 5. Save Ground Truth (on success)

If ALL code blocks execute successfully, ask:

> "All code executed successfully. Save as ground truth? [Y/n]"

If yes, append to `benchmarks/skills/<skill-name>/ground_truth.yaml`.
See [ground-truth-format.md](references/ground-truth-format.md) for the YAML schema.

### 6. Handle Failure

If any code block fails, show the error and ask:

> "Code execution failed. [R]etry with fix, [S]kip, or [A]bort?"

- **Retry**: Generate fixed code and re-execute
- **Skip**: Continue without saving
- **Abort**: End the session

## Notes

- Create `benchmarks/skills/<skill-name>/` directory if it doesn't exist
- Auto-increment ground truth IDs (gt-001, gt-002, etc.)
- Tag examples with inferred categories (e.g., "evaluation", "scorer", "dataset")
