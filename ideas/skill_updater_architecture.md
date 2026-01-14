# Skill Updater: Architecture Design

System for updating Claude Code skills from evolving documentation sources.

---

## Executive Summary

**Goal**: Enable skills like `mlflow-evaluation` to stay current as documentation evolves, through a semi-automated workflow that merges new documentation content while preserving curated quality.

**Architecture**: `skill-updater` skill + `/update-skill` command combo
**Philosophy**: AI-assisted, human-approved (dry-run by default)

---

## 1. Core Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Update trigger | Manual invocation | User controls when updates happen |
| Merge strategy | Always merge, flag conflicts | Preserves curation while surfacing decisions |
| Source tracking | Section metadata + central manifest | Granular citations + bulk operations |
| Conflict resolution | All to human | Never auto-resolve ambiguity |
| Review interface | Interactive CLI (section-level) | Balances speed and control |
| Validation | Syntax + execution + benchmarks | Three-layer safety net |
| Deprecation | Move to archive | Preserves history without cluttering active content |
| Dry run | Default behavior | Changes only applied with explicit confirmation |

---

## 2. Directory Structure

```
.claude/skills/
├── skill-updater/                    # The meta-skill itself
│   ├── SKILL.md                      # Entry point with workflows
│   └── references/
│       ├── SCHEMAS.md                # Data schemas for skills/manifests
│       ├── MERGE-PATTERNS.md         # Example merges (before/after)
│       ├── EXTRACTION-GUIDE.md       # How to extract from web docs
│       └── VALIDATION-GUIDE.md       # Validation procedures
│
├── mlflow-evaluation/                # Target skill (example)
│   ├── SKILL.md
│   ├── references/
│   │   ├── GOTCHAS.md
│   │   ├── patterns-*.md
│   │   └── deprecated/              # Archive for deprecated content
│   │       └── patterns-legacy.md
│   └── meta/                        # NEW: Metadata directory
│       ├── sources.yaml             # Source manifest
│       ├── snapshots/               # Version snapshots
│       │   └── 2024-01-15.tar.gz
│       └── curation-stats.yaml      # Human vs AI content ratio
```

---

## 3. Data Schemas

### 3.1 Source Manifest (`meta/sources.yaml`)

```yaml
version: 1
skill: mlflow-evaluation
last_updated: 2024-01-15T10:30:00Z

sources:
  - url: https://docs.databricks.com/en/mlflow/genai-evaluation.html
    target_files:
      - CRITICAL-interfaces.md
      - patterns-evaluation.md
    last_fetched: 2024-01-15T10:30:00Z
    content_hash: sha256:abc123...
    status: active

  - url: https://mlflow.org/docs/latest/llms/tracing/index.html
    target_files:
      - patterns-trace-analysis.md
    last_fetched: 2024-01-10T08:00:00Z
    content_hash: sha256:def456...
    status: active

pending_sources: []  # URLs queued for integration
```

### 3.2 Section Metadata (Embedded in .md files)

```markdown
## Pattern 5: Evaluation with Ground Truth

> **Source**: https://docs.databricks.com/en/mlflow/genai-evaluation.html#ground-truth
> **Updated**: 2024-01-15 | **Origin**: AI-merged | **Reviewed**: pending

\```python
# Pattern code here...
\```
```

### 3.3 Curation Stats (`meta/curation-stats.yaml`)

```yaml
total_sections: 127
human_curated: 85
ai_generated: 32
ai_merged: 10
pending_review: 5

ratio_human: 0.67  # Alert threshold: 0.50
last_calculated: 2024-01-15T10:30:00Z
```

---

## 4. Command Interface

### `/update-skill` Command

```bash
# Basic usage (reads from manifest)
/update-skill mlflow-evaluation

# Add new URL to manifest and process
/update-skill mlflow-evaluation --add-url https://docs.databricks.com/... --target patterns-evaluation.md

# Process specific URLs without modifying manifest
/update-skill mlflow-evaluation --urls https://example.com/doc1,https://example.com/doc2

# Apply changes after review (exit dry-run)
/update-skill mlflow-evaluation --apply

# Bootstrap manifest for existing skill
/update-skill mlflow-evaluation --bootstrap
```

### Command Flags

| Flag | Purpose |
|------|---------|
| `--add-url URL --target FILE` | Add source to manifest |
| `--urls URL1,URL2` | Process URLs without manifest changes |
| `--apply` | Apply staged changes (exit dry-run) |
| `--bootstrap` | AI proposes initial manifest for existing skill |
| `--validate-only` | Run validation without merging |
| `--skip-benchmarks` | Skip benchmark regression tests |

---

## 5. Workflow Pipeline

### 5.1 High-Level Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FETCH     │────▶│   EXTRACT   │────▶│   MERGE     │────▶│  VALIDATE   │
│ (Web docs)  │     │ (AI filter) │     │ (AI + human)│     │ (3 layers)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                              ┌────────────────────┴───────────────────┐
                                              ▼                                        ▼
                                     ┌─────────────┐                          ┌─────────────┐
                                     │   STAGING   │──────(user confirms)────▶│   APPLY     │
                                     │ (dry-run)   │                          │ (commit)    │
                                     └─────────────┘                          └─────────────┘
```

### 5.2 Detailed Steps

#### Step 1: Fetch
- Read manifest to get source URLs
- Fetch each URL (skip auth-gated content)
- Convert to markdown
- Store in temp staging area

#### Step 2: Extract (Two-Pass for Large Docs)
- **Pass 1**: AI summarizes entire doc, identifies relevant sections
- **Pass 2**: AI deep-dives on relevant sections only
- Output: Extracted content chunks mapped to target files

#### Step 3: Semantic Diff
- Compare extracted content against current skill content
- Identify: additions, modifications, deletions, deprecations
- Flag meaningful changes (ignore typo fixes unless configured)

#### Step 4: Merge
- AI proposes merge for each change
- Group by section for human review
- Track provenance (human vs AI content)

#### Step 5: Conflict Resolution (Interactive CLI)
```
╔══════════════════════════════════════════════════════════════╗
║  CONFLICT: patterns-evaluation.md > Pattern 5                ║
╠══════════════════════════════════════════════════════════════╣
║  CURRENT (human-curated):                                    ║
║  ┌──────────────────────────────────────────────────────────┐║
║  │ results = mlflow.genai.evaluate(                         │║
║  │     data=dataset,                                        │║
║  │     scorers=[Correctness()]                              │║
║  │ )                                                        │║
║  └──────────────────────────────────────────────────────────┘║
║                                                              ║
║  NEW (from docs.databricks.com):                             ║
║  ┌──────────────────────────────────────────────────────────┐║
║  │ results = mlflow.genai.evaluate(                         │║
║  │     data=dataset,                                        │║
║  │     scorers=[Correctness()],                             │║
║  │     model_id="my-model"  # NEW PARAMETER                 │║
║  │ )                                                        │║
║  └──────────────────────────────────────────────────────────┘║
║                                                              ║
║  [1] Keep current                                            ║
║  [2] Accept new                                              ║
║  [3] Merge (add model_id as optional)  <-- AI suggested      ║
║  [4] Edit manually                                           ║
╚══════════════════════════════════════════════════════════════╝
Choice [1-4]:
```

#### Step 6: Validate (Three Layers)

| Layer | What | How | Failure Action |
|-------|------|-----|----------------|
| Syntax | Python blocks parse | AST parsing | Block merge |
| Execution | Examples run | Configurable env (mock/local/Databricks) | Flag for review |
| Benchmark | Quality gates pass | Run skill benchmarks | Show diff, block merge |

#### Step 7: Stage & Apply
- All changes staged (dry-run default)
- Show complete diff summary
- User confirms to apply
- On apply: update files, snapshot old version, update manifest

---

## 6. Quality Safeguards

### 6.1 Curation Ratio Tracking
- Every section tagged with origin: `human-curated`, `ai-generated`, `ai-merged`
- Alert when `human_curated / total` drops below 50%
- Dashboard shows drift over time

### 6.2 Golden Test Cases
- Maintain hand-verified test cases in `meta/golden-tests.yaml`
- These MUST pass regardless of other validation
- Catches semantic regressions that syntax/benchmarks miss

### 6.3 Semantic Review
- After merge, AI reviews entire updated file
- Flags: logical inconsistencies, contradictory patterns, suspicious changes
- Output: review report with confidence scores

### 6.4 Diff Review Mandate
- Every merge produces a diff
- Human MUST review diff before final commit
- Diff includes: source URLs, affected sections, validation results

---

## 7. Deprecation Handling

When documentation marks something deprecated:

1. **Detect**: AI identifies deprecation language in source docs
2. **Flag**: Surface to user with reason from source
3. **Archive**: Move content to `references/deprecated/` with metadata:
   ```markdown
   <!-- DEPRECATED: 2024-01-15 -->
   <!-- Reason: Replaced by mlflow.genai.evaluate() in MLflow 3.0 -->
   <!-- Successor: patterns-evaluation.md#pattern-1 -->

   ## Legacy Pattern: mlflow.evaluate()
   ...
   ```
4. **Cross-reference**: Add note in active content pointing to deprecated version for users on old MLflow

---

## 8. Snapshot & Rollback

### Snapshot Creation
- Before any update, create snapshot: `meta/snapshots/YYYY-MM-DD-HHMMSS.tar.gz`
- Contains: all reference files + manifest + curation stats
- Retention: keep last 10 snapshots

### Rollback Command
```bash
/update-skill mlflow-evaluation --rollback 2024-01-10-103000
```

---

## 9. skill-updater Skill Structure

The meta-skill that powers this system:

### SKILL.md
```markdown
---
name: skill-updater
description: |
  Update Claude Code skills from documentation sources. Use when:
  - Adding new documentation URLs to a skill
  - Refreshing skill content from updated docs
  - Bootstrapping source tracking for existing skills
  - Validating skill content against sources
---

## Before Using This Skill

Read these references in order:
1. SCHEMAS.md - Understand data structures
2. MERGE-PATTERNS.md - See merge examples
3. EXTRACTION-GUIDE.md - How content is extracted
4. VALIDATION-GUIDE.md - Validation procedures

## Workflows

### Workflow 1: Bootstrap Existing Skill
[Steps to create initial manifest via AI inference + human correction]

### Workflow 2: Add New Documentation Source
[Steps to add URL, specify target files, run initial sync]

### Workflow 3: Routine Update
[Steps to fetch all sources, merge changes, validate, apply]

### Workflow 4: Handle Conflicts
[Guide for interactive conflict resolution]

### Workflow 5: Rollback Update
[Steps to restore from snapshot]
```

### Reference Files
- **SCHEMAS.md**: YAML schemas for manifest, curation stats, section metadata
- **MERGE-PATTERNS.md**: Before/after examples for each merge type
- **EXTRACTION-GUIDE.md**: How to handle large docs, auth-gated content, etc.
- **VALIDATION-GUIDE.md**: Configuring test environments, golden tests, benchmarks

---

## 10. Implementation Phases

### Phase 1: Scaffold (Immediate)
- [ ] Create `skill-updater/` directory structure
- [ ] Write SKILL.md with workflows (stub content)
- [ ] Create reference file stubs
- [ ] Create `/update-skill` command file
- [ ] Define YAML schemas

### Phase 2: Core Pipeline
- [ ] Implement URL fetching + markdown conversion
- [ ] Implement AI extraction (two-pass)
- [ ] Implement semantic diff detection
- [ ] Implement basic merge logic

### Phase 3: Interactive CLI
- [ ] Build conflict presentation UI
- [ ] Implement selection + edit resolution
- [ ] Add staging area management
- [ ] Implement dry-run / apply flow

### Phase 4: Validation
- [ ] Syntax validation (AST parsing)
- [ ] Example execution framework
- [ ] Benchmark integration
- [ ] Golden test case support

### Phase 5: Quality & Polish
- [ ] Curation ratio tracking
- [ ] Semantic review post-merge
- [ ] Snapshot/rollback support
- [ ] Deprecation detection + archival

---

## 11. Critical Files to Create

| File | Purpose |
|------|---------|
| `.claude/skills/skill-updater/SKILL.md` | Main skill entry point |
| `.claude/skills/skill-updater/references/SCHEMAS.md` | Data schemas |
| `.claude/skills/skill-updater/references/MERGE-PATTERNS.md` | Merge examples |
| `.claude/skills/skill-updater/references/EXTRACTION-GUIDE.md` | Extraction logic |
| `.claude/skills/skill-updater/references/VALIDATION-GUIDE.md` | Validation procedures |
| `.claude/commands/update-skill.md` | Command definition |
| `.claude/skills/mlflow-evaluation/meta/sources.yaml` | Example manifest |

---

## 12. Verification Plan

After implementation, verify by:

1. **Bootstrap test**: Run `--bootstrap` on mlflow-evaluation, verify AI proposes reasonable sources
2. **Add URL test**: Add a known Databricks doc URL, verify extraction works
3. **Conflict test**: Manually create a conflict, verify interactive CLI presents it correctly
4. **Validation test**: Introduce a syntax error in merged content, verify it's caught
5. **Rollback test**: Make an update, rollback, verify state restored
6. **Benchmark test**: Run skill benchmarks before/after update, verify diff reporting works

---

## Additional Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Benchmark integration | Tight integration with `benchmarks/skills/` | Reuse existing infrastructure, avoid duplication |
| Batch updates | No `--all` flag | Always require explicit skill name to avoid accidental mass updates |
