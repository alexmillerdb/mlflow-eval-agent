"""Skill index and recommendation engine for proactive skill injection.

This module provides utilities for parsing skill metadata and recommending
relevant skills based on user queries. Skills are loaded from .claude/skills/
and their SKILL.md frontmatter is parsed to build a searchable index.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SkillMeta:
    """Parsed metadata from a skill's SKILL.md frontmatter.

    Attributes:
        name: Skill identifier (e.g., "mlflow-evaluation")
        description: Full description from frontmatter
        path: Path to the skill directory
        keywords: Extracted keywords for matching
        references: List of reference file paths
    """

    name: str
    description: str
    path: Path
    keywords: list[str] = field(default_factory=list)
    references: list[Path] = field(default_factory=list)


def parse_frontmatter(content: str) -> dict[str, str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Full markdown file content

    Returns:
        Dictionary of frontmatter key-value pairs
    """
    # Match YAML frontmatter between --- delimiters
    match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return {}

    frontmatter = {}
    for line in match.group(1).split('\n'):
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            frontmatter[key.strip()] = value.strip()

    return frontmatter


def extract_keywords(description: str) -> list[str]:
    """Extract searchable keywords from skill description.

    Args:
        description: Skill description text

    Returns:
        List of lowercase keywords
    """
    # Words that indicate when to use the skill
    trigger_patterns = [
        r'Use when ([^.]+)',
        r'Use for ([^.]+)',
        r'\((\d+)\) ([^,)]+)',
    ]

    keywords = set()

    # Extract trigger phrases
    for pattern in trigger_patterns:
        matches = re.findall(pattern, description, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = ' '.join(match)
            words = re.findall(r'\b[a-z]{3,}\b', match.lower())
            keywords.update(words)

    # Add common terms from description
    common_terms = re.findall(r'\b(?:evaluat|scor|trac|context|prompt|token|rag|dataset|debug|optim|latenc|error|failur)[a-z]*\b', description.lower())
    keywords.update(common_terms)

    return list(keywords)


def parse_skill_index(skills_dir: Optional[Path] = None) -> dict[str, SkillMeta]:
    """Parse SKILL.md frontmatter for all skills in directory.

    Args:
        skills_dir: Path to skills directory (default: .claude/skills/)

    Returns:
        Dictionary mapping skill names to SkillMeta objects
    """
    if skills_dir is None:
        # Default to .claude/skills/ relative to project root
        project_root = Path(__file__).parent.parent
        skills_dir = project_root / ".claude" / "skills"

    skills: dict[str, SkillMeta] = {}

    if not skills_dir.exists():
        return skills

    for skill_path in skills_dir.iterdir():
        if not skill_path.is_dir():
            continue

        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            continue

        try:
            content = skill_md.read_text()
            meta = parse_frontmatter(content)

            if "name" not in meta:
                continue

            # Find reference files
            references = []
            refs_dir = skill_path / "references"
            if refs_dir.exists():
                references = list(refs_dir.glob("*.md"))

            skill = SkillMeta(
                name=meta["name"],
                description=meta.get("description", ""),
                path=skill_path,
                keywords=extract_keywords(meta.get("description", "")),
                references=references,
            )
            skills[skill.name] = skill

        except Exception:
            # Skip malformed skill files
            continue

    return skills


# Keyword mapping for skill recommendations
SKILL_KEYWORDS = {
    "mlflow-evaluation": [
        "evaluate", "evaluation", "scorer", "scoring", "dataset", "test",
        "genai", "mlflow.genai", "correctness", "safety", "guidelines",
        "retrieval", "groundedness", "relevance", "judge", "llm judge",
    ],
    "trace-analysis": [
        "trace", "traces", "debug", "debugging", "latency", "performance",
        "span", "spans", "profil", "error", "failure", "architecture",
        "tool call", "rag", "retriev", "production", "logs",
    ],
    "context-engineering": [
        "context", "prompt", "system prompt", "token", "tokens", "budget",
        "few-shot", "example", "rag context", "format", "rot", "state",
        "conversation", "memory", "truncat", "overflow", "window",
    ],
}


def recommend_skills(query: str, skills_index: Optional[dict[str, SkillMeta]] = None) -> list[str]:
    """Recommend skills based on query keywords.

    Args:
        query: User query or task description
        skills_index: Parsed skill index (loaded if not provided)

    Returns:
        List of recommended skill names, ordered by relevance
    """
    query_lower = query.lower()
    scores: dict[str, int] = {}

    for skill_name, keywords in SKILL_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in query_lower:
                # Exact match gets higher score
                score += 2
            elif any(word in query_lower for word in keyword.split()):
                # Partial match
                score += 1

        if score > 0:
            scores[skill_name] = score

    # Sort by score descending
    sorted_skills = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return sorted_skills


def get_skill_gotchas(skill_name: str, skills_dir: Optional[Path] = None) -> Optional[str]:
    """Get GOTCHAS.md content for a skill if it exists.

    Args:
        skill_name: Name of the skill
        skills_dir: Path to skills directory

    Returns:
        Content of GOTCHAS.md or None if not found
    """
    if skills_dir is None:
        project_root = Path(__file__).parent.parent
        skills_dir = project_root / ".claude" / "skills"

    gotchas_path = skills_dir / skill_name / "references" / "GOTCHAS.md"
    if gotchas_path.exists():
        return gotchas_path.read_text()

    return None


def get_skill_references(skill_name: str, skills_dir: Optional[Path] = None) -> dict[str, Path]:
    """Get all reference files for a skill.

    Args:
        skill_name: Name of the skill
        skills_dir: Path to skills directory

    Returns:
        Dictionary mapping reference names to file paths
    """
    if skills_dir is None:
        project_root = Path(__file__).parent.parent
        skills_dir = project_root / ".claude" / "skills"

    refs_dir = skills_dir / skill_name / "references"
    if not refs_dir.exists():
        return {}

    return {ref.stem: ref for ref in refs_dir.glob("*.md")}


def format_skill_summary(skills_index: dict[str, SkillMeta]) -> str:
    """Format skill index as a summary string for injection into prompts.

    Args:
        skills_index: Parsed skill index

    Returns:
        Formatted string summarizing available skills
    """
    if not skills_index:
        return "No skills available."

    lines = ["## Available Skills\n"]

    for name, meta in skills_index.items():
        # Truncate description for summary
        desc = meta.description[:150] + "..." if len(meta.description) > 150 else meta.description
        ref_count = len(meta.references)

        lines.append(f"### {name}")
        lines.append(f"{desc}")
        if ref_count > 0:
            lines.append(f"*{ref_count} reference files available*\n")
        else:
            lines.append("")

    return "\n".join(lines)


# Pre-load skill index at module import time for fast access
_SKILL_INDEX: Optional[dict[str, SkillMeta]] = None


def get_skill_index() -> dict[str, SkillMeta]:
    """Get the global skill index, loading it if necessary.

    Returns:
        Dictionary mapping skill names to SkillMeta objects
    """
    global _SKILL_INDEX
    if _SKILL_INDEX is None:
        _SKILL_INDEX = parse_skill_index()
    return _SKILL_INDEX
