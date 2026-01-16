"""
Scorers for mlflow-evaluation skill benchmark.

Two-tier scoring strategy:
- Tier 1: Pattern-matching scorers (deterministic API correctness checks)
- Tier 3: Guidelines scorers (subjective quality assessment)
"""

import ast
import os
import re
from mlflow.genai.scorers import scorer, Guidelines, Safety
from mlflow.entities import Feedback


# =============================================================================
# Helper Functions
# =============================================================================

def get_response_text(outputs) -> str:
    """Extract response text from outputs.

    Handles both formats:
    - String: outputs = "response text" (MLflow standard)
    - Dict: outputs = {"response": "text"} (legacy format)
    """
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, dict):
        return str(outputs.get("response", ""))
    return str(outputs) if outputs else ""


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
    response = get_response_text(outputs)
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
    response = get_response_text(outputs)
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
    response = get_response_text(outputs)
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
    response = get_response_text(outputs)
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
    response = get_response_text(outputs)
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


@scorer
def predict_fn_unpacks_kwargs(outputs) -> Feedback:
    """
    Check if predict_fn signature receives unpacked kwargs.

    CORRECT:   def my_app(query, context=None)
    WRONG:     def my_app(inputs)
    """
    response = get_response_text(outputs)
    code = extract_all_code(response)

    if not code or "predict_fn" not in response.lower():
        return Feedback(
            name="predict_fn_unpacks_kwargs",
            value="skip",
            rationale="No predict_fn found in response"
        )

    # Look for predict_fn definition or reference
    # Check for common mistake: single 'inputs' parameter
    has_inputs_param = bool(re.search(r'def\s+\w+\s*\(\s*inputs\s*[,\)]', code))

    # Check for correct pattern: multiple named params
    has_unpacked = bool(re.search(r'def\s+\w+\s*\([^)]*\bquery\b', code)) or \
                   bool(re.search(r'def\s+\w+\s*\([^)]*\bprompt\b', code)) or \
                   bool(re.search(r'def\s+\w+\s*\(\*\*', code))  # **kwargs

    if has_inputs_param and not has_unpacked:
        return Feedback(
            name="predict_fn_unpacks_kwargs",
            value="no",
            rationale="predict_fn uses 'inputs' parameter instead of unpacking kwargs"
        )

    if has_unpacked:
        return Feedback(
            name="predict_fn_unpacks_kwargs",
            value="yes",
            rationale="predict_fn correctly unpacks kwargs"
        )

    return Feedback(
        name="predict_fn_unpacks_kwargs",
        value="skip",
        rationale="Could not determine predict_fn signature pattern"
    )


@scorer
def guidelines_has_name(outputs) -> Feedback:
    """
    Check if Guidelines scorers include required 'name' parameter.

    CORRECT:   Guidelines(name="check", guidelines="...")
    WRONG:     Guidelines(guidelines="...")
    """
    response = get_response_text(outputs)
    code = extract_all_code(response)

    if not code or "Guidelines" not in code:
        return Feedback(
            name="guidelines_has_name",
            value="skip",
            rationale="No Guidelines scorer found"
        )

    # Find all Guidelines() calls
    guidelines_calls = re.findall(r'Guidelines\s*\([^)]+\)', code, re.DOTALL)

    if not guidelines_calls:
        return Feedback(
            name="guidelines_has_name",
            value="skip",
            rationale="No Guidelines() instantiation found"
        )

    missing_name = []
    for i, call in enumerate(guidelines_calls):
        if "name=" not in call and "name =" not in call:
            missing_name.append(f"call {i+1}")

    if missing_name:
        return Feedback(
            name="guidelines_has_name",
            value="no",
            rationale=f"Guidelines() calls missing 'name' parameter: {', '.join(missing_name)}"
        )

    return Feedback(
        name="guidelines_has_name",
        value="yes",
        rationale=f"All {len(guidelines_calls)} Guidelines calls include 'name' parameter"
    )


@scorer
def returns_valid_feedback(outputs) -> Feedback:
    """
    Check if scorers return valid types: Feedback, bool, int, float, str.

    CORRECT:   return Feedback(value=True, rationale="...")
    CORRECT:   return True, return 0.85, return "yes"
    WRONG:     return {"score": 0.5}
    """
    response = get_response_text(outputs)
    code = extract_all_code(response)

    if not code:
        return Feedback(
            name="returns_valid_feedback",
            value="skip",
            rationale="No code blocks found"
        )

    # Look for custom scorer definitions
    scorer_funcs = re.findall(
        r'@scorer.*?def\s+(\w+)\s*\([^)]*\).*?return\s+(.+?)(?:\n|$)',
        code,
        re.DOTALL
    )

    if not scorer_funcs:
        return Feedback(
            name="returns_valid_feedback",
            value="skip",
            rationale="No custom scorers with return statements found"
        )

    invalid_returns = []
    for func_name, return_expr in scorer_funcs:
        # Check for dict return (common mistake)
        if re.search(r'\{[^}]*["\']score["\']', return_expr):
            invalid_returns.append(f"{func_name}: returns dict")

    if invalid_returns:
        return Feedback(
            name="returns_valid_feedback",
            value="no",
            rationale=f"Invalid return types found: {'; '.join(invalid_returns)}"
        )

    return Feedback(
        name="returns_valid_feedback",
        value="yes",
        rationale=f"All {len(scorer_funcs)} scorers return valid types"
    )


@scorer
def trace_search_uses_attributes(outputs) -> Feedback:
    """
    Check if trace search uses 'attributes.' prefix in filter strings.

    CORRECT:   mlflow.search_traces("attributes.status = 'OK'")
    WRONG:     mlflow.search_traces("status = 'OK'")
    """
    response = get_response_text(outputs)
    code = extract_all_code(response)

    if not code or "search_traces" not in code:
        return Feedback(
            name="trace_search_uses_attributes",
            value="skip",
            rationale="No search_traces() call found"
        )

    # Find search_traces calls with filter strings
    search_calls = re.findall(
        r'search_traces\s*\([^)]*["\']([^"\']+)["\'][^)]*\)',
        code
    )

    if not search_calls:
        return Feedback(
            name="trace_search_uses_attributes",
            value="skip",
            rationale="No search_traces() filter strings found"
        )

    missing_prefix = []
    for i, filter_str in enumerate(search_calls):
        # Check for common attribute names without 'attributes.' prefix
        if re.search(r'\b(status|span_type|name)\s*=', filter_str) and \
           'attributes.' not in filter_str:
            missing_prefix.append(f"call {i+1}: '{filter_str[:50]}'")

    if missing_prefix:
        return Feedback(
            name="trace_search_uses_attributes",
            value="no",
            rationale=f"search_traces filters missing 'attributes.' prefix: {'; '.join(missing_prefix)}"
        )

    return Feedback(
        name="trace_search_uses_attributes",
        value="yes",
        rationale=f"All {len(search_calls)} search_traces calls use proper 'attributes.' prefix"
    )


@scorer(aggregations=["mean", "min", "max"])
def code_block_count(outputs) -> int:
    """Count the number of Python code blocks in the response."""
    response = get_response_text(outputs)
    blocks = extract_code_blocks(response)
    return len(blocks)




# =============================================================================
# Tier 3: Guidelines Scorers (Subjective Quality) - Lazy Factory
# =============================================================================

# Guidelines text constants (for reuse)
# NOTE: Guidelines scorer returns yes/no/na - criteria are lenient to avoid false negatives
CODE_QUALITY_GUIDELINES = """
Evaluate if the code is clean and well-organized.

Rate 'yes' if the code:
- Is syntactically valid Python
- Uses reasonably descriptive variable/function names
- Has appropriate structure (functions or classes where helpful)
- Includes at least basic docstrings or comments

Rate 'no' only if:
- Code has syntax errors
- Code is severely disorganized or unreadable
- Names are completely unclear (single letters everywhere)

Most working code with docstrings should pass. Don't penalize for minor style issues.
"""

RESPONSE_COMPLETENESS_GUIDELINES = """
Evaluate if the response addresses the user's request with usable code.

Rate 'yes' if:
- Code addresses the user's request
- Code is syntactically complete (has imports, no placeholders like ... or TODO)
- Code has docstrings or comments explaining what it does

Rate 'no' only if:
- Code has placeholders (... or TODO) instead of actual implementation
- Missing critical imports that would prevent the code from running
- Doesn't solve the requested task

IMPORTANT: Don't penalize for:
- Missing installation instructions (pip install)
- Not including extensive tutorials or usage guides
- Using simplified example data
- Not being "production-grade"
- Using hardcoded values for demonstration purposes

Most complete code examples should pass.
"""

EXPLANATION_CLARITY_GUIDELINES = """
Evaluate if the response explains the code adequately.

Rate 'yes' if:
- Code has docstrings explaining what functions do
- The response includes any explanation of what the code does
- The overall intent is clear from reading the code and comments

Rate 'no' only if:
- No explanation whatsoever (no docstrings, no comments, no text)
- Explanation is confusing or contradicts the code
- Critical information is missing that makes the code incomprehensible

IMPORTANT: Code-heavy responses don't need extensive prose. Docstrings and
inline comments count as explanation. Self-documenting code with clear
function names is acceptable.

Most responses with docstrings should pass.
"""


def _create_tier3_scorers(model: str) -> list:
    """
    Create Tier 3 Guidelines scorers with configured model.

    Uses lenient yes/no criteria to avoid false negatives on good code.
    Guidelines explicitly state what NOT to penalize (e.g., missing install
    instructions, simplified example data, not being production-grade).

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
    predict_fn_unpacks_kwargs,
    guidelines_has_name,
    returns_valid_feedback,
    trace_search_uses_attributes,
    code_block_count,
]

def get_scorers(preset: str = "full", model: str = None) -> list:
    """
    Get a preset collection of scorers.

    Tier 1 scorers are deterministic (no LLM needed).
    Safety and Tier 3 scorers require an LLM and are created lazily with the specified model.

    Args:
        preset: One of "full", "quick", "tier1", "tier3"
        model: LLM model for Safety and Tier 3 scorers (default: from env or databricks:/databricks-gpt-5-2)

    Returns:
        List of scorer objects
    """
    model = model or DEFAULT_JUDGE_MODEL

    # Create Safety scorer with explicit model to avoid malformed URI
    # (MLflow default is just "databricks" when tracking_uri=databricks, which is invalid)
    safety_scorer = Safety(model=model)

    # Build scorer list based on preset
    if preset == "quick":
        return [safety_scorer] + TIER1_SCORERS

    if preset == "tier1":
        return TIER1_SCORERS

    if preset == "tier3":
        return _create_tier3_scorers(model)

    if preset in ("full", "all"):
        tier3 = _create_tier3_scorers(model)
        return [safety_scorer] + TIER1_SCORERS + tier3

    # Default to full
    tier3 = _create_tier3_scorers(model)
    return [safety_scorer] + TIER1_SCORERS + tier3


if __name__ == "__main__":
    print("MLflow Evaluation Skill - Benchmark Scorers")
    print("=" * 50)
    print(f"\nDefault Judge Model: {DEFAULT_JUDGE_MODEL}")

    print(f"\nTier 1 (Pattern-matching): {len(TIER1_SCORERS)} scorers")
    for s in TIER1_SCORERS:
        name = getattr(s, "name", getattr(s, "__name__", str(s)))
        print(f"  - {name}")

    print(f"\nTier 3 (Guidelines): Created lazily with model parameter")
    print("  - code_quality")
    print("  - response_completeness")
    print("  - explanation_clarity")

    print(f"\nPresets:")
    print("  quick: Safety + Tier 1 (no LLM needed)")
    print("  tier1: Tier 1 only")
    print("  tier3: Tier 3 only (requires LLM)")
    print("  full:  Safety + Tier 1 + Tier 3")
