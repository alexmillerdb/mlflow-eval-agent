"""
Scorers for mlflow-evaluation skill benchmark.

Three-tier scoring strategy:
- Tier 1: Pattern-matching scorers (deterministic checks)
- Tier 2: make_judge with curated references (API correctness grounded in docs)
- Tier 3: Guidelines scorers (subjective quality assessment)
"""

import ast
import os
import re
from mlflow.genai.scorers import scorer, Guidelines, Safety
from mlflow.genai.judges import make_judge
from mlflow.entities import Feedback


# =============================================================================
# Helper Functions
# =============================================================================

def extract_code_blocks(response: str) -> list[str]:
    """Extract Python code blocks from markdown response."""
    # Match ```python ... ``` blocks
    pattern = r'```python\s*(.*?)```'
    blocks = re.findall(pattern, response, re.DOTALL)

    # Also match ``` ... ``` blocks that look like Python
    generic_pattern = r'```\s*(.*?)```'
    generic_blocks = re.findall(generic_pattern, response, re.DOTALL)

    for block in generic_blocks:
        if block not in blocks and ('import ' in block or 'def ' in block or '@' in block):
            blocks.append(block)

    return blocks


def extract_all_code(response: str) -> str:
    """Extract and concatenate all code blocks from response."""
    blocks = extract_code_blocks(response)
    return "\n\n".join(blocks)


# =============================================================================
# Tier 1: Pattern-Matching Scorers (Deterministic)
# =============================================================================

@scorer
def code_syntax_valid(outputs) -> Feedback:
    """Check if all Python code blocks in the response have valid syntax."""
    response = str(outputs.get("response", ""))
    blocks = extract_code_blocks(response)

    if not blocks:
        return Feedback(
            name="code_syntax_valid",
            value="skip",
            rationale="No Python code blocks found in response"
        )

    invalid_blocks = []
    for i, block in enumerate(blocks):
        try:
            ast.parse(block)
        except SyntaxError as e:
            invalid_blocks.append(f"Block {i+1}: {e.msg} at line {e.lineno}")

    if invalid_blocks:
        return Feedback(
            name="code_syntax_valid",
            value="no",
            rationale=f"Syntax errors found: {'; '.join(invalid_blocks)}"
        )

    return Feedback(
        name="code_syntax_valid",
        value="yes",
        rationale=f"All {len(blocks)} code block(s) have valid Python syntax"
    )


@scorer
def uses_genai_evaluate(outputs) -> Feedback:
    """Check if code uses mlflow.genai.evaluate() (not deprecated mlflow.evaluate())."""
    response = str(outputs.get("response", ""))
    code = extract_all_code(response)

    if not code:
        return Feedback(
            name="uses_genai_evaluate",
            value="skip",
            rationale="No code blocks found"
        )

    has_correct = "mlflow.genai.evaluate" in code
    # Check for deprecated pattern: mlflow.evaluate() without genai
    has_deprecated = bool(re.search(r'mlflow\.evaluate\s*\(', code)) and "mlflow.genai" not in code

    if has_deprecated:
        return Feedback(
            name="uses_genai_evaluate",
            value="no",
            rationale="Uses deprecated mlflow.evaluate() instead of mlflow.genai.evaluate()"
        )

    if has_correct:
        return Feedback(
            name="uses_genai_evaluate",
            value="yes",
            rationale="Correctly uses mlflow.genai.evaluate()"
        )

    return Feedback(
        name="uses_genai_evaluate",
        value="skip",
        rationale="No evaluate() call found (may not be an evaluation script)"
    )


@scorer
def has_nested_inputs(outputs) -> Feedback:
    """Check if evaluation data uses proper nested {"inputs": {...}} structure."""
    response = str(outputs.get("response", ""))
    code = extract_all_code(response)

    if not code:
        return Feedback(
            name="has_nested_inputs",
            value="skip",
            rationale="No code blocks found"
        )

    # Check for correct pattern: "inputs": { or 'inputs': {
    has_nested = bool(re.search(r'["\']inputs["\']\s*:\s*\{', code))

    # Check for flat structure indicators (common mistakes)
    flat_patterns = [
        r'eval_data\s*=\s*\[\s*\{[^}]*["\']query["\']',  # Direct query key
        r'data\s*=\s*\[\s*\{[^}]*["\']prompt["\']',  # Direct prompt key
    ]
    has_flat = any(re.search(p, code) for p in flat_patterns)

    # If has nested structure, it's correct
    if has_nested:
        return Feedback(
            name="has_nested_inputs",
            value="yes",
            rationale="Data uses proper nested {'inputs': {...}} structure"
        )

    # If has flat patterns without nested, it's wrong
    if has_flat and not has_nested:
        return Feedback(
            name="has_nested_inputs",
            value="no",
            rationale="Data appears to use flat structure instead of nested {'inputs': {...}}"
        )

    return Feedback(
        name="has_nested_inputs",
        value="skip",
        rationale="Could not determine data structure pattern"
    )


@scorer
def has_scorer_decorator(outputs) -> Feedback:
    """Check if custom scorers use the @scorer decorator."""
    response = str(outputs.get("response", ""))
    code = extract_all_code(response)

    if not code:
        return Feedback(
            name="has_scorer_decorator",
            value="skip",
            rationale="No code blocks found"
        )

    # Find function definitions that look like scorers
    scorer_funcs = re.findall(r'def\s+(\w+)\s*\([^)]*(?:inputs|outputs|trace)[^)]*\)', code)

    if not scorer_funcs:
        return Feedback(
            name="has_scorer_decorator",
            value="skip",
            rationale="No custom scorer functions found"
        )

    # Check if @scorer decorator is present
    has_decorator = "@scorer" in code

    if has_decorator:
        return Feedback(
            name="has_scorer_decorator",
            value="yes",
            rationale=f"Found @scorer decorator for custom scorer functions"
        )

    return Feedback(
        name="has_scorer_decorator",
        value="no",
        rationale=f"Custom scorer function(s) found ({', '.join(scorer_funcs)}) but missing @scorer decorator"
    )


@scorer
def uses_valid_aggregations(outputs) -> Feedback:
    """Check if aggregations use valid names (min, max, mean, median, variance, p90)."""
    response = str(outputs.get("response", ""))
    code = extract_all_code(response)

    if not code:
        return Feedback(
            name="uses_valid_aggregations",
            value="skip",
            rationale="No code blocks found"
        )

    # Check for aggregations parameter
    agg_match = re.search(r'aggregations\s*=\s*\[(.*?)\]', code, re.DOTALL)

    if not agg_match:
        return Feedback(
            name="uses_valid_aggregations",
            value="skip",
            rationale="No aggregations parameter found"
        )

    agg_content = agg_match.group(1)
    valid_aggs = {"min", "max", "mean", "median", "variance", "p90"}
    invalid_aggs = {"p50", "p99", "sum", "avg", "count"}

    found_invalid = []
    for invalid in invalid_aggs:
        if f'"{invalid}"' in agg_content or f"'{invalid}'" in agg_content:
            found_invalid.append(invalid)

    if found_invalid:
        return Feedback(
            name="uses_valid_aggregations",
            value="no",
            rationale=f"Invalid aggregation(s): {found_invalid}. Valid: min, max, mean, median, variance, p90"
        )

    return Feedback(
        name="uses_valid_aggregations",
        value="yes",
        rationale="All aggregations use valid names"
    )


@scorer(aggregations=["mean", "min", "max"])
def code_block_count(outputs) -> int:
    """Count the number of Python code blocks in the response."""
    response = str(outputs.get("response", ""))
    blocks = extract_code_blocks(response)
    return len(blocks)


# =============================================================================
# Tier 2: make_judge with Curated References (Lazy Factory)
# =============================================================================

# Key gotchas extracted from GOTCHAS.md for make_judge grounding
KEY_GOTCHAS = """
MLflow 3 GenAI API Critical Requirements:

1. EVALUATE FUNCTION: Use mlflow.genai.evaluate() NOT mlflow.evaluate()
   - mlflow.evaluate() is deprecated for GenAI evaluation

2. DATA STRUCTURE: Must use nested format with 'inputs' key
   - CORRECT: {"inputs": {"query": "..."}}
   - WRONG: {"query": "..."} (flat structure)

3. PREDICT_FN SIGNATURE: Receives unpacked kwargs, not a dict
   - CORRECT: def my_app(query, context=None)
   - WRONG: def my_app(inputs)

4. SCORER DECORATOR: Custom scorers must use @scorer decorator
   - CORRECT: @scorer def my_scorer(outputs): ...
   - WRONG: def my_scorer(outputs): ... (missing decorator)

5. GUIDELINES SCORER: Requires both 'name' and 'guidelines' parameters
   - CORRECT: Guidelines(name="check", guidelines="...")
   - WRONG: Guidelines(guidelines="...") (missing name)

6. FEEDBACK RETURNS: Scorers return Feedback objects or primitives
   - CORRECT: return Feedback(value=True, rationale="...")
   - CORRECT: return True, return 0.85, return "yes"
   - WRONG: return {"score": 0.5} (dict not allowed)

7. AGGREGATIONS: Only these 6 are valid: min, max, mean, median, variance, p90
   - WRONG: p50, p99, sum (not valid)

8. TRACE SEARCH: Use attributes. prefix and single quotes
   - CORRECT: mlflow.search_traces("attributes.status = 'OK'")
   - WRONG: mlflow.search_traces("status = 'OK'") (missing prefix)
"""


def _create_tier2_scorers(model: str) -> list:
    """
    Create Tier 2 make_judge scorers with configured model.

    Args:
        model: Model endpoint (e.g., "databricks:/databricks-gpt-5-2")

    Returns:
        List of make_judge scorer instances
    """
    api_correctness_judge = make_judge(
        name="api_correctness",
        model=model,
        instructions=f"""
You are evaluating if generated Python code follows MLflow 3 GenAI best practices.

Reference Documentation (CRITICAL - use this as ground truth):
{KEY_GOTCHAS}

Code to evaluate:
{{{{ outputs }}}}

Evaluation criteria:
1. Check if the code avoids the common mistakes listed above
2. Verify correct API usage patterns
3. Look for any deprecated or incorrect patterns

Return 'yes' if the code follows best practices, 'no' if any mistakes are found.
Provide a brief rationale explaining your assessment.
"""
    )
    return [api_correctness_judge]


# =============================================================================
# Tier 3: Guidelines Scorers (Subjective Quality) - Lazy Factory
# =============================================================================

# Guidelines text constants (for reuse)
CODE_QUALITY_GUIDELINES = """
Evaluate the quality of the generated code:

1. COMPLETENESS: Code is complete and runnable (no placeholders like '...' or 'TODO')
2. IMPORTS: All necessary imports are included at the top of the code
3. NAMING: Variable and function names are descriptive and follow Python conventions
4. STRUCTURE: Code is well-organized with appropriate functions/classes
5. DOCUMENTATION: Functions have docstrings or comments where helpful

Rate as 'yes' if the code meets most of these criteria, 'no' otherwise.
"""

RESPONSE_COMPLETENESS_GUIDELINES = """
Evaluate if the response fully addresses the user's request:

1. ADDRESSES REQUEST: The response directly answers what was asked
2. WORKING EXAMPLE: Includes a complete, working code example
3. EXPLANATION: Code is accompanied by helpful explanation or context
4. BEST PRACTICES: Follows recommended patterns for the task

Rate as 'yes' if the response is complete and helpful, 'no' otherwise.
"""

EXPLANATION_CLARITY_GUIDELINES = """
Evaluate the clarity of explanations in the response:

1. CLEAR LANGUAGE: Uses clear, concise language
2. APPROPRIATE DETAIL: Provides enough detail without being overwhelming
3. LOGICAL FLOW: Information is presented in a logical order
4. HIGHLIGHTS IMPORTANT: Key points or gotchas are clearly highlighted

Rate as 'yes' if explanations are clear and helpful, 'no' otherwise.
"""


def _create_tier3_scorers(model: str) -> list:
    """
    Create Tier 3 Guidelines scorers with configured model.

    Args:
        model: Model endpoint (e.g., "databricks:/databricks-gpt-5-2")

    Returns:
        List of Guidelines scorer instances
    """
    return [
        Guidelines(name="code_quality", guidelines=CODE_QUALITY_GUIDELINES, model=model),
        Guidelines(name="response_completeness", guidelines=RESPONSE_COMPLETENESS_GUIDELINES, model=model),
        Guidelines(name="explanation_clarity", guidelines=EXPLANATION_CLARITY_GUIDELINES, model=model),
    ]


# =============================================================================
# Scorer Collections
# =============================================================================

# Default model for LLM-based scorers
DEFAULT_JUDGE_MODEL = os.environ.get("BENCHMARK_JUDGE_MODEL", "databricks:/databricks-gpt-5-2")

# Tier 1: Deterministic pattern-matching scorers (no LLM needed)
TIER1_SCORERS = [
    code_syntax_valid,
    uses_genai_evaluate,
    has_nested_inputs,
    has_scorer_decorator,
    uses_valid_aggregations,
    code_block_count,
]

# Safety scorer (always include)
SAFETY_SCORER = Safety()


def get_scorers(preset: str = "full", model: str = None) -> list:
    """
    Get a preset collection of scorers.

    Tier 1 scorers are deterministic (no LLM needed).
    Tier 2 and 3 scorers require an LLM and are created lazily with the specified model.

    Args:
        preset: One of "full", "quick", "tier1", "tier2", "tier3"
        model: LLM model for Tier 2/3 scorers (default: from env or databricks:/databricks-gpt-5-2)

    Returns:
        List of scorer objects
    """
    model = model or DEFAULT_JUDGE_MODEL

    # Build scorer list based on preset
    if preset == "quick":
        return [SAFETY_SCORER] + TIER1_SCORERS

    if preset == "tier1":
        return TIER1_SCORERS

    if preset == "tier2":
        return _create_tier2_scorers(model)

    if preset == "tier3":
        return _create_tier3_scorers(model)

    if preset in ("full", "all"):
        tier2 = _create_tier2_scorers(model)
        tier3 = _create_tier3_scorers(model)
        return [SAFETY_SCORER] + TIER1_SCORERS + tier2 + tier3

    # Default to full
    tier2 = _create_tier2_scorers(model)
    tier3 = _create_tier3_scorers(model)
    return [SAFETY_SCORER] + TIER1_SCORERS + tier2 + tier3


if __name__ == "__main__":
    print("MLflow Evaluation Skill - Benchmark Scorers")
    print("=" * 50)
    print(f"\nDefault Judge Model: {DEFAULT_JUDGE_MODEL}")

    print(f"\nTier 1 (Pattern-matching): {len(TIER1_SCORERS)} scorers")
    for s in TIER1_SCORERS:
        name = getattr(s, "name", getattr(s, "__name__", str(s)))
        print(f"  - {name}")

    print(f"\nTier 2 (make_judge): Created lazily with model parameter")
    print("  - api_correctness")

    print(f"\nTier 3 (Guidelines): Created lazily with model parameter")
    print("  - code_quality")
    print("  - response_completeness")
    print("  - explanation_clarity")

    print(f"\nPresets:")
    print("  quick: Safety + Tier 1 (no LLM needed)")
    print("  tier1: Tier 1 only")
    print("  tier2: Tier 2 only (requires LLM)")
    print("  tier3: Tier 3 only (requires LLM)")
    print("  full:  Safety + Tier 1 + Tier 2 + Tier 3")
