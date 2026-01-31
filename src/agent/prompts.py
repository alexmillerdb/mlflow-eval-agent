"""Prompt loading utilities for the MLflow Evaluation Agent.

Handles loading prompts from both installed packages and development mode.
"""

import importlib.resources
import logging
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)


def _get_prompts_dir() -> Path:
    """Get prompts directory, works both installed and in dev.

    When installed as a wheel, uses importlib.resources.
    In development, falls back to relative path.
    """
    # Try package resources first (installed wheel)
    try:
        files = importlib.resources.files("prompts")
        # Check if it's a real directory we can use
        with importlib.resources.as_file(files) as prompts_path:
            if prompts_path.is_dir():
                return prompts_path
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass

    # Fallback to relative path (development)
    # From src/agent/prompts.py, go up to project root
    return Path(__file__).parent.parent.parent / "prompts"


@mlflow.trace
def load_prompt(name: str = "system") -> str:
    """Load external prompt from prompts/ directory.

    Works both when installed as a wheel and in development mode.
    """
    # Try importlib.resources first for installed package
    try:
        files = importlib.resources.files("prompts")
        prompt_file = files.joinpath(f"{name}.md")
        return prompt_file.read_text()
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass

    # Fallback to relative path (development)
    path = Path(__file__).parent.parent.parent / "prompts" / f"{name}.md"
    if path.exists():
        return path.read_text()

    logger.warning(f"Prompt file not found: {name}.md")
    return ""
