"""Syntax-highlighted code viewer for generated evaluation files.

Displays:
- Tab per generated file (scorers.py, eval_dataset.py, run_eval.py)
- Read-only code view with syntax highlighting
- Download buttons for each file
"""

from pathlib import Path
from typing import Optional

import streamlit as st

from ..services.state_reader import list_generated_files


def render_code_viewer(session_dir: Optional[Path]) -> None:
    """Render code viewer for generated evaluation files.

    Args:
        session_dir: Session directory path (or None if no session).
    """
    if not session_dir:
        st.info("No active session. Start the agent to generate evaluation code.")
        return

    session_path = Path(session_dir) if isinstance(session_dir, str) else session_dir

    # Get generated files
    files = list_generated_files(session_path)

    if not files:
        st.info(
            "No generated files yet. "
            "The agent will create evaluation files during task execution."
        )
        return

    st.markdown("### Generated Evaluation Code")
    st.caption(f"Files location: `{session_path / 'evaluation'}`")

    # Create tabs for each file
    file_names = [f["name"] for f in files]
    tabs = st.tabs(file_names)

    for tab, file_info in zip(tabs, files):
        with tab:
            _render_file_content(file_info)


def _render_file_content(file_info: dict) -> None:
    """Render a single file with syntax highlighting and download.

    Args:
        file_info: File information dictionary.
    """
    content = file_info["content"]
    name = file_info["name"]
    path = file_info["path"]

    # File info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption(f"`{path}`")
    with col2:
        st.caption(f"{file_info['size']:,} bytes")
    with col3:
        st.download_button(
            label="Download",
            data=content,
            file_name=name,
            mime="text/x-python",
            key=f"download_{name}",
        )

    # Code display with syntax highlighting
    st.code(content, language="python", line_numbers=True)


def render_code_summary(session_dir: Optional[Path]) -> None:
    """Render a brief summary of generated code status.

    Useful for sidebar or overview display.

    Args:
        session_dir: Session directory path.
    """
    if not session_dir:
        st.caption("No generated files")
        return

    session_path = Path(session_dir) if isinstance(session_dir, str) else session_dir
    files = list_generated_files(session_path)

    if not files:
        st.caption("No generated files")
        return

    # Expected files
    expected = {"eval_dataset.py", "scorers.py", "run_eval.py"}
    found = {f["name"] for f in files}

    # Status badges
    for name in expected:
        if name in found:
            st.markdown(f":white_check_mark: `{name}`")
        else:
            st.markdown(f":x: `{name}` (missing)")

    # Extra files
    extra = found - expected
    if extra:
        st.caption("Additional files:")
        for name in extra:
            st.markdown(f":page_facing_up: `{name}`")


def render_run_instructions(session_dir: Optional[Path]) -> None:
    """Render instructions for running the generated evaluation.

    Args:
        session_dir: Session directory path.
    """
    if not session_dir:
        return

    session_path = Path(session_dir) if isinstance(session_dir, str) else session_dir
    run_eval_path = session_path / "evaluation" / "run_eval.py"

    if not run_eval_path.exists():
        return

    st.markdown("### Running the Evaluation")
    st.markdown("Execute the generated evaluation script:")
    st.code(f"python {run_eval_path}", language="bash")

    st.markdown("Or import and run programmatically:")
    st.code(
        f"""import sys
sys.path.insert(0, "{session_path / 'evaluation'}")
from run_eval import run_evaluation

results = run_evaluation()
print(results)""",
        language="python",
    )
