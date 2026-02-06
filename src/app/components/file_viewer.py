"""File viewer component for inspecting session artifacts.

Displays generated evaluation scripts, analysis JSON, task definitions,
and other session files with syntax highlighting.
"""

from pathlib import Path
from typing import Dict, List

import streamlit as st

# Maximum file size to display (1 MB)
_MAX_FILE_SIZE = 1_000_000

# Language mapping by extension
_LANG_MAP = {
    ".py": "python",
    ".json": "json",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".txt": "text",
}


def _discover_files(session_dir: Path) -> Dict[str, Path]:
    """Scan session directory for known artifact files.

    Returns:
        Dict mapping display names to file paths, ordered by priority.
    """
    files: Dict[str, Path] = {}

    # Known top-level files
    for name in ("eval_tasks.json",):
        path = session_dir / name
        if path.is_file():
            files[name] = path

    # State directory
    state_dir = session_dir / "state"
    if state_dir.is_dir():
        analysis = state_dir / "analysis.json"
        if analysis.is_file():
            files["state/analysis.json"] = analysis

    # Evaluation scripts
    eval_dir = session_dir / "evaluation"
    if eval_dir.is_dir():
        for py_file in sorted(eval_dir.glob("*.py")):
            files[f"evaluation/{py_file.name}"] = py_file

    # Any other JSON files in root (excluding already-added ones)
    for json_file in sorted(session_dir.glob("*.json")):
        display_name = json_file.name
        if display_name not in files:
            files[display_name] = json_file

    return files


def _format_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def render_file_viewer(session_dir: Path) -> None:
    """Render an expander-based file viewer for session artifacts.

    Args:
        session_dir: Path to the agent session directory.
    """
    if not session_dir or not session_dir.is_dir():
        st.info("No session directory available. Run the agent to generate files.")
        return

    files = _discover_files(session_dir)

    if not files:
        st.info(f"No files found in {session_dir}")
        return

    # Find most recently modified file to auto-expand
    newest_file = max(files.keys(), key=lambda n: files[n].stat().st_mtime)

    for name, path in files.items():
        is_newest = (name == newest_file)
        with st.expander(f"📄 {name}", expanded=is_newest):
            size = path.stat().st_size
            st.caption(f"{_format_size(size)}")

            lang = _LANG_MAP.get(path.suffix, "text")

            if size > _MAX_FILE_SIZE:
                st.warning(
                    f"File is {_format_size(size)} (exceeds 1 MB limit). "
                    "Showing first 1 MB."
                )
                content = path.read_text(errors="replace")[:_MAX_FILE_SIZE]
            else:
                content = path.read_text(errors="replace")

            st.code(content, language=lang, line_numbers=True)
