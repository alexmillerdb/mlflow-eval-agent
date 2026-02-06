# MLflow Eval Agent - Databricks Apps Implementation Spec

> **Purpose**: Phased implementation plan for restructuring codebase and adding Streamlit-based Databricks Apps deployment. Designed for modular execution by Claude Code sub-agents.

**Created**: 2024-01-30
**Status**: Phase 4 Complete
**Current Phase**: Ready for Phase 5

---

## Quick Reference

| Phase | Name | Status | Features |
|-------|------|--------|----------|
| 0 | Codebase Restructure | ✅ Complete | 0.1-0.11 |
| 1 | Streamlit MVP | ✅ Complete | 1.1-1.6 |
| 2 | Files SDK Integration | ✅ Complete | 2.1-2.6 |
| 3 | Runtime & Auth | ✅ Complete | 3.1-3.4 |
| 4 | Enhanced UI | ✅ Complete | 4.1-4.4 |
| 5 | Deployment | ⬜ Not Started | 5.1-5.4 |

---

## Architecture Overview

### Target Structure

```
src/
├── __init__.py
├── cli.py                        # Root CLI (wheel entry point)
│
├── core/                         # SHARED INFRASTRUCTURE
│   ├── __init__.py              # Exports: Config, RuntimeContext, etc.
│   ├── config.py                # Environment configuration
│   ├── runtime.py               # Runtime detection (LOCAL/JOB/APP)
│   ├── auth.py                  # Unified Databricks auth
│   └── files.py                 # Databricks Files SDK wrapper
│
├── agent/                        # AUTONOMOUS AGENT LOGIC
│   ├── __init__.py              # Exports: MLflowAgent, run_autonomous
│   ├── agent.py                 # MLflowAgent class
│   ├── tools.py                 # MCP tools (MLflow + Files)
│   ├── mlflow_ops.py            # MLflow operations + state
│   ├── prompts.py               # Prompt loading
│   ├── autonomous.py            # run_autonomous() loop
│   └── skills.py                # Skill definitions
│
└── app/                          # STREAMLIT UI
    ├── __init__.py
    ├── main.py                   # Streamlit entry point
    ├── streaming.py              # Async→sync bridge
    └── components/
        ├── __init__.py
        ├── chat.py               # Chat interface
        ├── sidebar.py            # Config sidebar
        └── progress.py           # Task progress
```

### Dependency Flow

```
cli.py (wheel) ──► agent/ ──► core/
                              ▲
app/ (Streamlit) ─────────────┘
```

### Deployment Modes

| Mode | Package | Entry Point | Use Case |
|------|---------|-------------|----------|
| Jobs | Wheel | `mlflow-eval` CLI | Scheduled/batch evaluation |
| Apps | Source | `streamlit run src/app/main.py` | Interactive UI |

---

## Phase 0: Codebase Restructure ✅

**Status**: Complete (2026-01-30)
**Commit**: `a6c945c`

Reorganized flat `src/` into `core/` + `agent/` + `app/` hierarchy.

[Full details → docs/archive/phase-0-restructure.md](docs/archive/phase-0-restructure.md)

---

## Phase 1: Streamlit MVP ✅

**Status**: Complete (2026-02-02)
**Branch**: `fix-env-vars`

Basic Streamlit chat interface with Claude Agent SDK streaming via async-to-sync bridge.

[Full details → docs/archive/phase-1-streamlit-mvp.md](docs/archive/phase-1-streamlit-mvp.md)

---

## Phase 2: Files SDK Integration ✅

**Status**: Complete (2026-02-02)
**Branch**: `fix-env-vars`

Added MCP tools for UC Volumes and Workspace Files (6 new tools, 9 total). Includes unit tests (29) and integration tests (7).

[Full details → docs/archive/phase-2-files-sdk.md](docs/archive/phase-2-files-sdk.md)

---

## Phase 3: Runtime & Auth ✅

**Status**: Complete (2026-02-02)
**Branch**: `fix-env-vars`

Added DATABRICKS_APP runtime context detection and OAuth authentication helpers for Streamlit apps.

[Full details → docs/archive/phase-3-runtime-auth.md](docs/archive/phase-3-runtime-auth.md)

---

## Phase 4: Enhanced UI ✅

**Status**: Complete (2026-02-02)
**Branch**: `fix-env-vars`

Full-featured Streamlit UI with tabbed interface (Interactive/Autonomous modes), progress components, and Playwright E2E tests.

[Full details → docs/archive/phase-4-enhanced-ui.md](docs/archive/phase-4-enhanced-ui.md)

---

## Phase 5: Deployment

**Goal**: Deploy to Databricks Apps via DABs
**Estimated Effort**: Small
**Dependencies**: Phases 0-4 complete
**Risk**: Deployment configuration

### Feature 5.1: Create App Configuration

**Status**: ⬜ Not Started

**Task**: Create Databricks App configuration file.

**Create `app.yaml`**:
```yaml
# Databricks App Configuration
# https://docs.databricks.com/dev-tools/databricks-apps/

name: mlflow-eval-agent
description: Interactive MLflow trace analysis and evaluation agent

# Entry point command
command:
  - streamlit
  - run
  - src/app/main.py

# Environment variables
env:
  - name: MLFLOW_TRACKING_URI
    value: databricks
  - name: MLFLOW_AGENT_VOLUME_PATH
    valueFrom:
      config: volume_path
  - name: ANTHROPIC_MODEL
    valueFrom:
      config: model

# App-level configuration (set at deploy time)
config:
  - name: volume_path
    description: Unity Catalog Volume path for session storage
    default: /Volumes/users/default/mlflow-eval-agent
  - name: experiment_id
    description: Default MLflow experiment ID
    default: ""
  - name: anthropic_model
    description: Claude model to use
    default: databricks-claude-opus-4.5
```

---

### Feature 5.2: Extend DABs Configuration

**Status**: ⬜ Not Started

**Task**: Add apps section to databricks.yml.

**Update `databricks.yml`** - Add under resources:
```yaml
resources:
  # Existing jobs section...

  apps:
    eval_agent_app:
      name: "[${bundle.target}] MLflow Eval Agent App"
      description: "Streamlit UI for MLflow trace analysis and evaluation"
      source_code_path: .
      config:
        - name: volume_path
          value: "${var.volume_path}"
        - name: experiment_id
          value: "${var.experiment_id}"
        - name: model
          value: "${var.anthropic_model}"
```

---

### Feature 5.3: Test Deployment

**Status**: ⬜ Not Started

**Task**: Validate and deploy to Databricks.

**Commands**:
```bash
# Validate bundle configuration
databricks bundle validate -t dev

# Deploy (builds wheel and creates app)
databricks bundle deploy -t dev

# List apps to verify
databricks apps list

# Get app URL
databricks apps get mlflow-eval-agent
```

**Verification Checklist**:
- [ ] Bundle validates without errors
- [ ] App deploys successfully
- [ ] App URL is accessible
- [ ] OAuth login works
- [ ] Can query MLflow experiments
- [ ] Streaming responses work

---

### Feature 5.4: Full Deployment Validation

**Status**: ⬜ Not Started

**Task**: Comprehensive deployment validation before production release.

**Level 1: Bundle Validation**:
```bash
databricks bundle validate -t dev
```

**Level 2: Deploy & Verify**:
```bash
databricks bundle deploy -t dev
databricks apps list
databricks apps get mlflow-eval-agent
```

**Level 3: App Functional Testing**:
1. Open app URL from `databricks apps get` output
2. Verify OAuth login works
3. Submit test query
4. Verify streaming response
5. Check UC Volume for session files

**Level 4: End-to-End**:
1. Run autonomous mode from deployed app
2. Verify tasks complete
3. Run generated eval script
4. Check MLflow for all traces

**Success Criteria**:
- [ ] App accessible via Databricks URL
- [ ] OAuth authenticates correctly
- [ ] Full agent functionality works
- [ ] Session state persists to UC Volume

---

## Execution Guide

### Sub-Agent Strategy

| Phase | Recommended Approach |
|-------|---------------------|
| 0 (Restructure) | Single agent, sequential features |
| 1 (Streamlit) | Single agent |
| 2 (Files SDK) | Parallel: 2.1-2.3 can run together |
| 3 (Runtime) | Single agent, small scope |
| 4 (Enhanced UI) | Sequential, depends on prior phases |
| 5 (Deployment) | Single agent |

### Testing Strategy

**Testing Ladder** (Applied to Each Phase):

```
Level 1: Unit Tests (pytest)        → Fast, no external deps
Level 2: Import Verification        → Catches circular imports
Level 3: Mock Integration (--mock)  → Tests logic without MLflow
Level 4: Local Integration          → Real MLflow, local environment
Level 5: Autonomous Run             → Full agent loop locally
Level 6: Databricks Deployment      → Production validation (Phase 0 + Phase 5 only)
```

**After each feature**:
1. Run specific unit tests
2. Run import verification
3. Run full test suite for regressions

**After each phase**:
1. Run `/test-phase <phase-number>` for comprehensive validation
2. Manual verification where required
3. Update status in this file

**Before shipping**:
1. Run `/test-full-cycle <experiment-id>` for complete validation
2. Run `/validate-deployment --target dev` for deployment testing

### Error Debugging Workflow

When tests fail, use this pattern:

1. **Capture trace ID** from test output
2. **Analyze trace**: `/analyze-trace <trace-id>`
   - Check span errors
   - Identify bottlenecks
   - Review tool call sequence
3. **Analyze tokens**: `/analyze-tokens <trace-id>`
   - Check cache efficiency (>50% = good)
   - Identify context growth issues
4. **Fix issue** based on analysis
5. **Re-run same test** to verify fix
6. **Proceed** to next feature/phase

**Key Metrics to Watch**:

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Cache efficiency | >50% | 30-50% | <30% |
| Initializer cost | <$1.50 | $1.50-$2.50 | >$2.50 |
| Worker cost | <$0.75 | $0.75-$1.25 | >$1.25 |
| Context growth | Stable | Slow growth | Runaway |

### Progress Tracking

Update feature status using:
- ⬜ Not Started
- 🔄 In Progress
- ✅ Complete
- ❌ Blocked

### Phase Archiving

When a phase is complete, archive it to keep this spec focused on remaining work:

1. **Create archive file**: `docs/archive/phase-N-name.md`
   - Copy full phase content (goal, features, verification steps)
   - Mark success criteria as checked
   - Add completion date and commit hash

2. **Update this spec**: Replace phase details with summary:
   ```markdown
   ## Phase N: Name ✅

   **Status**: Complete (YYYY-MM-DD)
   **Commit**: `abc1234`

   One-line description of what was accomplished.

   [Full details → docs/archive/phase-N-name.md](docs/archive/phase-N-name.md)
   ```

3. **Verify**: Ensure relative link works and main spec is under ~1,600 lines

---

## Appendix: File Reference

### Files by Phase

**Phase 0 (Restructure)**:
- `src/core/__init__.py`
- `src/core/config.py` (moved)
- `src/core/runtime.py` (moved)
- `src/core/auth.py` (moved + renamed)
- `src/agent/__init__.py`
- `src/agent/agent.py` (moved + modified)
- `src/agent/autonomous.py` (new)
- `src/agent/prompts.py` (new)
- `src/agent/tools.py` (moved)
- `src/agent/mlflow_ops.py` (moved)
- `src/cli.py` (modified imports)

**Phase 1 (Streamlit)**:
- `src/app/__init__.py`
- `src/app/main.py`
- `src/app/streaming.py`
- `pyproject.toml` (add streamlit)

**Phase 2 (Files SDK)**:
- `src/core/files.py` (new)
- `src/agent/tools.py` (add file tools)

**Phase 3 (Runtime)**:
- `src/core/runtime.py` (add APP context)
- `src/app/auth.py` (new)

**Phase 4 (Enhanced UI)**:
- `src/app/components/progress.py`
- `src/app/components/sidebar.py`
- `src/app/main.py` (add tabs)

**Phase 5 (Deployment)**:
- `app.yaml` (new)
- `databricks.yml` (add apps section)
