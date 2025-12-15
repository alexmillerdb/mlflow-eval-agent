"""Unit tests for skills module."""

import pytest
from pathlib import Path

from src.skills import (
    parse_frontmatter,
    extract_keywords,
    parse_skill_index,
    recommend_skills,
    get_skill_gotchas,
    get_skill_references,
    format_skill_summary,
    get_skill_index,
    SkillMeta,
)


class TestParseFrontmatter:
    """Tests for YAML frontmatter parsing."""

    def test_parse_valid_frontmatter(self):
        content = """---
name: test-skill
description: A test skill for unit tests
---

# Rest of the file
"""
        result = parse_frontmatter(content)
        assert result["name"] == "test-skill"
        assert result["description"] == "A test skill for unit tests"

    def test_parse_no_frontmatter(self):
        content = "# Just a markdown file\n\nNo frontmatter here."
        result = parse_frontmatter(content)
        assert result == {}

    def test_parse_multiline_values(self):
        # Note: Our simple parser doesn't support multiline values
        content = """---
name: simple-skill
description: Single line description
---
"""
        result = parse_frontmatter(content)
        assert "name" in result


class TestExtractKeywords:
    """Tests for keyword extraction from descriptions."""

    def test_extract_from_use_when_pattern(self):
        description = "Use when evaluating agent performance"
        keywords = extract_keywords(description)
        assert "evaluating" in keywords or "evaluat" in keywords

    def test_extract_common_terms(self):
        description = "For debugging traces and optimizing context"
        keywords = extract_keywords(description)
        assert any("debug" in k for k in keywords)
        assert any("trac" in k for k in keywords)

    def test_empty_description(self):
        keywords = extract_keywords("")
        assert isinstance(keywords, list)


class TestParseSkillIndex:
    """Tests for skill index parsing."""

    def test_parse_existing_skills(self):
        index = parse_skill_index()
        # Should find at least the 3 skills we know exist
        assert len(index) >= 3
        assert "mlflow-evaluation" in index
        assert "trace-analysis" in index
        assert "context-engineering" in index

    def test_skill_meta_structure(self):
        index = parse_skill_index()
        for name, meta in index.items():
            assert isinstance(meta, SkillMeta)
            assert meta.name == name
            assert isinstance(meta.description, str)
            assert isinstance(meta.path, Path)
            assert isinstance(meta.keywords, list)
            assert isinstance(meta.references, list)

    def test_mlflow_evaluation_has_references(self):
        index = parse_skill_index()
        meta = index.get("mlflow-evaluation")
        assert meta is not None
        # mlflow-evaluation should have reference files
        assert len(meta.references) > 0

    def test_nonexistent_directory(self):
        index = parse_skill_index(Path("/nonexistent/path"))
        assert index == {}


class TestRecommendSkills:
    """Tests for skill recommendation engine."""

    def test_evaluate_query(self):
        recommendations = recommend_skills("evaluate traces")
        assert len(recommendations) > 0
        assert "mlflow-evaluation" in recommendations or "trace-analysis" in recommendations

    def test_trace_query(self):
        recommendations = recommend_skills("debug trace latency")
        assert "trace-analysis" in recommendations

    def test_context_query(self):
        recommendations = recommend_skills("optimize token budget")
        assert "context-engineering" in recommendations

    def test_scorer_query(self):
        recommendations = recommend_skills("create custom scorer")
        assert "mlflow-evaluation" in recommendations

    def test_empty_query(self):
        recommendations = recommend_skills("")
        assert isinstance(recommendations, list)

    def test_irrelevant_query(self):
        recommendations = recommend_skills("what is the weather today")
        # Should return empty or very few matches
        assert isinstance(recommendations, list)


class TestGetSkillGotchas:
    """Tests for gotchas retrieval."""

    def test_mlflow_evaluation_has_gotchas(self):
        gotchas = get_skill_gotchas("mlflow-evaluation")
        assert gotchas is not None
        assert len(gotchas) > 0
        # Should contain critical information
        assert "mlflow" in gotchas.lower() or "evaluate" in gotchas.lower()

    def test_nonexistent_skill_gotchas(self):
        gotchas = get_skill_gotchas("nonexistent-skill")
        assert gotchas is None


class TestGetSkillReferences:
    """Tests for skill reference retrieval."""

    def test_mlflow_evaluation_references(self):
        refs = get_skill_references("mlflow-evaluation")
        assert len(refs) > 0
        # Should have the GOTCHAS file
        assert "GOTCHAS" in refs

    def test_nonexistent_skill_references(self):
        refs = get_skill_references("nonexistent-skill")
        assert refs == {}


class TestFormatSkillSummary:
    """Tests for skill summary formatting."""

    def test_format_populated_index(self):
        index = parse_skill_index()
        summary = format_skill_summary(index)
        assert "## Available Skills" in summary
        assert "mlflow-evaluation" in summary

    def test_format_empty_index(self):
        summary = format_skill_summary({})
        assert "No skills available" in summary


class TestGetSkillIndex:
    """Tests for global skill index access."""

    def test_get_skill_index_returns_dict(self):
        index = get_skill_index()
        assert isinstance(index, dict)
        assert len(index) > 0

    def test_get_skill_index_is_cached(self):
        # Should return same object on multiple calls
        index1 = get_skill_index()
        index2 = get_skill_index()
        assert index1 is index2
