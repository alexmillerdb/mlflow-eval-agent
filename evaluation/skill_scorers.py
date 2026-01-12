"""
Reusable scorers for skill benchmarks.

These scorers can be used across different skill benchmarks for common
code quality and response quality checks.
"""

import ast
import re
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback


# =============================================================================
# Code Block Extraction Helpers
# =============================================================================

def extract_code_blocks(response: str, language: str = "python") -> list[str]:
    """
    Extract code blocks from markdown response.

    Args:
        response: The response text containing markdown code blocks
        language: Language to look for (default: python)

    Returns:
        List of code block contents
    """
    # Match ```language ... ``` blocks
    pattern = rf'```{language}\s*(.*?)```'
    blocks = re.findall(pattern, response, re.DOTALL)

    # Also match generic ``` ... ``` blocks that look like the language
    if language == "python":
        generic_pattern = r'```\s*(.*?)```'
        generic_blocks = re.findall(generic_pattern, response, re.DOTALL)

        for block in generic_blocks:
            if block not in blocks and ('import ' in block or 'def ' in block or 'class ' in block):
                blocks.append(block)

    return blocks


def extract_all_code(response: str, language: str = "python") -> str:
    """Extract and concatenate all code blocks from response."""
    blocks = extract_code_blocks(response, language)
    return "\n\n".join(blocks)


# =============================================================================
# Syntax and Structure Scorers
# =============================================================================

@scorer
def code_syntax_valid(outputs) -> Feedback:
    """
    Check if all Python code blocks in the response have valid syntax.

    Uses ast.parse() to validate Python syntax.
    """
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
def code_has_imports(outputs) -> Feedback:
    """Check if code blocks contain import statements."""
    response = str(outputs.get("response", ""))
    code = extract_all_code(response)

    if not code:
        return Feedback(
            name="code_has_imports",
            value="skip",
            rationale="No code blocks found"
        )

    has_imports = bool(re.search(r'^(?:import|from)\s+\w+', code, re.MULTILINE))

    return Feedback(
        name="code_has_imports",
        value="yes" if has_imports else "no",
        rationale="Code includes import statements" if has_imports else "No import statements found"
    )


@scorer
def code_has_functions(outputs) -> Feedback:
    """Check if code blocks define functions."""
    response = str(outputs.get("response", ""))
    code = extract_all_code(response)

    if not code:
        return Feedback(
            name="code_has_functions",
            value="skip",
            rationale="No code blocks found"
        )

    func_count = len(re.findall(r'^def\s+\w+\s*\(', code, re.MULTILINE))

    if func_count > 0:
        return Feedback(
            name="code_has_functions",
            value="yes",
            rationale=f"Code defines {func_count} function(s)"
        )

    return Feedback(
        name="code_has_functions",
        value="no",
        rationale="No function definitions found"
    )


@scorer
def code_has_docstrings(outputs) -> Feedback:
    """Check if functions have docstrings."""
    response = str(outputs.get("response", ""))
    blocks = extract_code_blocks(response)

    if not blocks:
        return Feedback(
            name="code_has_docstrings",
            value="skip",
            rationale="No code blocks found"
        )

    total_funcs = 0
    funcs_with_docs = 0

    for block in blocks:
        try:
            tree = ast.parse(block)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_funcs += 1
                    if ast.get_docstring(node):
                        funcs_with_docs += 1
        except SyntaxError:
            continue

    if total_funcs == 0:
        return Feedback(
            name="code_has_docstrings",
            value="skip",
            rationale="No functions found"
        )

    ratio = funcs_with_docs / total_funcs
    has_most = ratio >= 0.5

    return Feedback(
        name="code_has_docstrings",
        value="yes" if has_most else "no",
        rationale=f"{funcs_with_docs}/{total_funcs} functions have docstrings ({ratio:.0%})"
    )


# =============================================================================
# Metrics Scorers
# =============================================================================

@scorer(aggregations=["mean", "min", "max"])
def code_block_count(outputs) -> int:
    """Count the number of Python code blocks in the response."""
    response = str(outputs.get("response", ""))
    blocks = extract_code_blocks(response)
    return len(blocks)


@scorer(aggregations=["mean", "min", "max", "median"])
def response_word_count(outputs) -> int:
    """Count words in the response."""
    response = str(outputs.get("response", ""))
    return len(response.split())


@scorer(aggregations=["mean", "min", "max"])
def code_line_count(outputs) -> int:
    """Count total lines of code across all code blocks."""
    response = str(outputs.get("response", ""))
    code = extract_all_code(response)
    if not code:
        return 0
    return len(code.strip().split('\n'))


# =============================================================================
# Pattern Checkers
# =============================================================================

@scorer
def no_placeholder_code(outputs) -> Feedback:
    """Check that code doesn't contain placeholders like '...' or 'TODO'."""
    response = str(outputs.get("response", ""))
    code = extract_all_code(response)

    if not code:
        return Feedback(
            name="no_placeholder_code",
            value="skip",
            rationale="No code blocks found"
        )

    placeholders = []

    # Check for ellipsis (but not in type hints)
    if re.search(r'(?<!:)\s*\.\.\.\s*(?!])', code):
        # Filter out legitimate uses like Protocol[...]
        if '...' in code and 'Protocol' not in code and 'Callable' not in code:
            placeholders.append("...")

    # Check for TODO/FIXME comments
    if re.search(r'#\s*(TODO|FIXME|XXX)', code, re.IGNORECASE):
        placeholders.append("TODO/FIXME comments")

    # Check for placeholder text
    placeholder_patterns = [
        r'your[_-]?api[_-]?key',
        r'your[_-]?token',
        r'<your[_-]',
        r'placeholder',
        r'add[_-]?code[_-]?here',
    ]
    for pattern in placeholder_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            placeholders.append(f"placeholder text matching '{pattern}'")

    if placeholders:
        return Feedback(
            name="no_placeholder_code",
            value="no",
            rationale=f"Found placeholders: {', '.join(placeholders)}"
        )

    return Feedback(
        name="no_placeholder_code",
        value="yes",
        rationale="No placeholder code found"
    )


# =============================================================================
# Scorer Collections
# =============================================================================

# All reusable scorers
ALL_SKILL_SCORERS = [
    code_syntax_valid,
    code_has_imports,
    code_has_functions,
    code_has_docstrings,
    code_block_count,
    response_word_count,
    code_line_count,
    no_placeholder_code,
]

# Quick syntax check only
SYNTAX_SCORERS = [
    code_syntax_valid,
    code_has_imports,
]

# Code quality scorers
CODE_QUALITY_SCORERS = [
    code_syntax_valid,
    code_has_imports,
    code_has_functions,
    code_has_docstrings,
    no_placeholder_code,
]

# Metrics only
METRICS_SCORERS = [
    code_block_count,
    response_word_count,
    code_line_count,
]


if __name__ == "__main__":
    print("Reusable Skill Scorers")
    print("=" * 50)
    print(f"\nAll scorers ({len(ALL_SKILL_SCORERS)}):")
    for s in ALL_SKILL_SCORERS:
        name = getattr(s, "__name__", str(s))
        print(f"  - {name}")
