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
# Databricks App Runtime Configuration
# https://docs.databricks.com/dev-tools/databricks-apps/app-runtime
# Valid top-level keys: command, env

command:
  - streamlit
  - run
  - src/app/main.py
  - --server.port
  - "8000"
  - --server.address
  - "0.0.0.0"

env:
  # MLflow tracking (auto-discovered by Databricks SDK)
  - name: MLFLOW_TRACKING_URI
    value: "databricks"

  # Target experiment (from experiment resource binding in databricks.yml)
  - name: MLFLOW_EXPERIMENT_ID
    valueFrom: "experiment-binding"

  # Agent's own traces experiment
  - name: MLFLOW_AGENT_EXPERIMENT_ID
    value: "${var.mlflow_agent_experiment_id}"

  # UC Volume path for session storage
  - name: MLFLOW_AGENT_VOLUME_PATH
    value: "${var.volume_path}"

  # Anthropic FM API config (served via Databricks endpoint)
  - name: ANTHROPIC_MODEL
    value: "${var.anthropic_model}"
  - name: ANTHROPIC_API_KEY
    value: ""
  - name: ANTHROPIC_AUTH_TOKEN
    valueFrom: "secrets-binding"
  - name: ANTHROPIC_BASE_URL
    value: "${var.anthropic_endpoint}"
  - name: ANTHROPIC_CUSTOM_HEADERS
    value: "x-databricks-use-coding-agent-mode: true"
```

**Notes on `app.yaml` schema**:
- Only `command` and `env` are valid top-level keys (no `config` section)
- `valueFrom` references a resource binding name from `databricks.yml` (not `config`)
- Databricks Apps auto-inject `DATABRICKS_CLIENT_ID`/`DATABRICKS_CLIENT_SECRET` for the app's service principal; the Databricks SDK auto-discovers these for MLflow, UC, etc.
- `ANTHROPIC_AUTH_TOKEN` is a PAT from a secret scope, used by `src/core/auth.py` to authenticate with the Anthropic FM API endpoint

---

### Feature 5.2: Extend DABs Configuration

**Status**: ⬜ Not Started

**Task**: Add apps section to databricks.yml.

**Update `databricks.yml`** - Add under resources:
```yaml
resources:
  # ... existing jobs section ...

  apps:
    eval_agent_app:
      name: "mlflow-eval-agent"
      description: "Streamlit UI for MLflow trace analysis and evaluation"
      source_code_path: .

      # Resource bindings grant the app's service principal access
      # and make values available to app.yaml via valueFrom
      resources:
        # Secret scope: Databricks PAT for Anthropic FM API auth
        - name: "secrets-binding"
          secret:
            scope: "${var.secret_scope}"
            key: "databricks-token"
            permission: "READ"

        # UC Volume: Session storage
        - name: "uc-volume-binding"
          uc_securable:
            securable_full_name: "users.alex_miller.mlflow-eval-agent"
            securable_type: "VOLUME"
            permission: "WRITE_VOLUME"

        # MLflow experiment: Target experiment to analyze
        - name: "experiment-binding"
          # Grants the SP read access to the experiment
```

**Supported resource binding types** (for reference):

| Type | Binding Field | Example Use |
|------|---------------|-------------|
| Secret | `secret: {scope, key, permission}` | PAT for FM API auth |
| UC Volume | `uc_securable: {securable_full_name, securable_type, permission}` | Session storage |
| Serving Endpoint | `serving_endpoint: {id, permission}` | Model serving |
| SQL Warehouse | `sql_warehouse: {id, permission}` | SQL queries |
| Job | `job: {id, permission}` | Job triggering |

**Auth flow**: The app's service principal gets `DATABRICKS_CLIENT_ID`/`DATABRICKS_CLIENT_SECRET` auto-injected. Resource bindings grant the SP additional permissions (secret read, volume write, experiment access). The `valueFrom` in `app.yaml` resolves secret values at runtime.

---

### Feature 5.2.1: Apps Storage Architecture (Local-First + Volume Sync)

**Status**: ⬜ Not Started

**Problem**: UC Volumes have no FUSE mount in Databricks Apps. `Path("/Volumes/...").write_text()` works on clusters and jobs (FUSE-mounted), but **fails in Apps** because `/Volumes/` doesn't exist on the local filesystem. The Files API (`src/core/files.py`) is the only way to access Volumes from Apps.

| Environment | `/Volumes/` access | `Path.write_text()` on Volume path |
|---|---|---|
| Clusters/Notebooks | FUSE mount | Works |
| Databricks Jobs | FUSE mount | Works |
| **Databricks Apps** | **Files API only** | **FAILS** |

**Solution**: Local-first writes with Volume sync at session boundaries. The agent writes session state to a local temp directory (fast, always works), then syncs to UC Volume via Files API at key checkpoints (durable, survives restarts).

- `MLFLOW_AGENT_VOLUME_PATH` controls the **sync destination**, not the working directory
- Session ID format changes from `"app-{app_name}"` (shared across users/runs) to `"app-{YYYYMMDD_HHMMSS}"` (unique per run)
- Local filesystem is writable but ephemeral — files vanish on app restart/redeployment
- Sync uses existing `uc_volume_write`/`uc_volume_read` from `src/core/files.py`

**Code changes required** (4 files):

**1. `src/core/runtime.py` — `get_sessions_base_path()`**

In APP mode, return a local temp path instead of the Volume path. The Volume path is used only for sync, not as the working directory.

```python
# Current (broken in Apps):
#   if volume_path:
#       base = Path(volume_path) / "sessions"  # FAILS — no FUSE mount
#
# Fixed:
#   if runtime.context == RuntimeContext.DATABRICKS_APP:
#       base = Path("/tmp/eval-agent/sessions")  # Local, always writable
#   elif volume_path:
#       base = Path(volume_path) / "sessions"    # FUSE mount (Jobs/clusters)
```

**2. `src/core/config.py` — Session ID generation**

Change APP session prefix from `"app-{app_name}"` (collision risk — shared across concurrent users) to timestamp-based `"app-{YYYYMMDD_HHMMSS}"` (unique per run).

```python
# In RuntimeInfo.session_prefix:
#   Current: return f"app-{self.app_name}"
#   Fixed:   return ""  (falls through to timestamp generation in Config.from_env)
#
# This makes APP mode use the same datetime.now().strftime("%Y-%m-%d_%H%M%S")
# logic as LOCAL mode, but with an "app-" prefix added in Config.from_env.
```

**3. `src/agent/mlflow_ops.py` — Add sync functions**

Add two functions that use the Files API to sync session state to/from UC Volume:

```python
def sync_session_to_volume(session_dir: Path, volume_path: str) -> None:
    """Sync local session directory to UC Volume via Files API.

    Walks session_dir and writes each file to the corresponding
    Volume path using uc_volume_write(). Called after each iteration.
    """
    ...

def restore_session_from_volume(session_id: str, volume_path: str, local_base: Path) -> Optional[Path]:
    """Restore a session from UC Volume to local filesystem.

    Uses uc_volume_read() + uc_volume_list() to download session files.
    Called on startup to resume interrupted sessions.
    Returns the local session_dir if restored, None if not found.
    """
    ...
```

**4. `src/agent/autonomous.py` — Call sync at session boundaries**

- On startup (APP mode): call `restore_session_from_volume()` to resume interrupted sessions
- After each iteration: call `sync_session_to_volume()` to persist state to Volume
- Only syncs when `MLFLOW_AGENT_VOLUME_PATH` is set and runtime is APP

```python
# In run_autonomous(), after set_session_dir():
#   if runtime.context == RuntimeContext.DATABRICKS_APP and volume_path:
#       restore_session_from_volume(config.session_id, volume_path, sessions_base)
#
# After each iteration (initializer/worker completes):
#   if runtime.context == RuntimeContext.DATABRICKS_APP and volume_path:
#       sync_session_to_volume(session_dir, volume_path)
```

**What does NOT change**:
- **Prompts** (`prompts/`) — use `{session_dir}` which resolves at runtime; storage is infrastructure-level
- **`src/core/files.py`** — already has the correct Files API implementation (`uc_volume_write`, `uc_volume_read`, `uc_volume_list`)
- **`src/agent/tools.py`** — MCP tools are unaffected; they operate on the resolved session directory

---

### Feature 5.2.2: Apps Authentication Architecture

**Status**: ⬜ Not Started

**Problem**: Three auth identities exist in Databricks Apps — service principal (SP), personal access token (PAT), and on-behalf-of (OBO) — and two code paths currently mix them up, which would cause runtime failures.

**Auth path diagram**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Databricks App Runtime                          │
│                                                                     │
│  ┌─── SP (auto-injected) ────────────────────────────────────────┐  │
│  │  DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET              │  │
│  │  → WorkspaceClient() (default auth)                           │  │
│  │  → MLflow tracking, UC Volumes, Workspace files               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─── PAT (from secret scope) ───────────────────────────────────┐  │
│  │  ANTHROPIC_AUTH_TOKEN (valueFrom: secrets-binding)             │  │
│  │  → ANTHROPIC_BASE_URL + Bearer header                         │  │
│  │  → Claude SDK → Anthropic FM API serving endpoint             │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─── OBO (per-request header) ──────────────────────────────────┐  │
│  │  x-forwarded-access-token (injected by Apps proxy)            │  │
│  │  → get_obo_token() → WorkspaceClient(token=obo)              │  │
│  │  → Per-user file ops, user identity (current_user.me())       │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Issue 1: PAT leaking into `DATABRICKS_TOKEN`**

`src/core/auth.py:configure_env()` (lines 37-42) unconditionally copies `ANTHROPIC_AUTH_TOKEN` into `DATABRICKS_TOKEN`. In Apps, this overrides the SP's OAuth M2M auth (`DATABRICKS_CLIENT_ID`/`SECRET`) with a PAT that lacks the SP's resource binding permissions. MLflow tracking, UC Volume writes, and Workspace file ops would all fail with permission errors.

**Fix**: Skip the `DATABRICKS_TOKEN` assignment when running as a Databricks App:

```python
# In src/core/auth.py, configure_env():
from .runtime import detect_runtime, RuntimeContext

runtime = detect_runtime()

# 2. Set DATABRICKS_TOKEN from ANTHROPIC_AUTH_TOKEN if available
# Skip in APP mode — SP uses CLIENT_ID/SECRET (OAuth M2M), not PAT
if runtime.context != RuntimeContext.DATABRICKS_APP:
    if not os.getenv("DATABRICKS_TOKEN"):
        if token := os.getenv("ANTHROPIC_AUTH_TOKEN"):
            os.environ["DATABRICKS_TOKEN"] = token
            configured["DATABRICKS_TOKEN"] = "***"
```

**Issue 2: Missing `DATABRICKS_HOST` in `app.yaml`**

`WorkspaceClientManager.get_user_client()` in `src/core/files.py:58` requires `DATABRICKS_HOST` to create OBO clients. In Apps, the Databricks SDK can auto-detect the host for the SP client, but OBO client creation happens before `configure_env()` runs (or may be called independently). Without `DATABRICKS_HOST` in the environment, OBO falls back to SP silently.

**Fix**: Add `DATABRICKS_HOST` to `app.yaml` env section and the corresponding variable to `databricks.yml`:

```yaml
# In app.yaml, add to env:
- name: DATABRICKS_HOST
  value: "${var.workspace_host}"
```

```yaml
# In databricks.yml, add to variables:
variables:
  workspace_host:
    description: "Databricks workspace URL (e.g., https://my-workspace.databricks.com)"
```

**What works correctly** (no changes needed):
- **OBO token extraction** (`src/app/auth.py:get_obo_token()`) — correctly reads `x-forwarded-access-token` from Streamlit headers
- **SP singleton** (`src/core/files.py:WorkspaceClientManager.get_service_client()`) — uses default `WorkspaceClient()` which auto-discovers SP credentials
- **Anthropic env vars** (`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY=""`) — correctly configured in `app.yaml`
- **OBO→SP fallback** (`WorkspaceClientManager.get_client()`) — returns SP client when no OBO token available

**Code changes summary**:

| File | Change |
|------|--------|
| `src/core/auth.py` | Guard PAT→TOKEN assignment with `runtime.context != DATABRICKS_APP` |
| `app.yaml` | Add `DATABRICKS_HOST` env var |
| `databricks.yml` | Add `workspace_host` variable |

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
- `databricks.yml` (add apps section + `workspace_host` variable)
- `src/core/runtime.py` (local-first path for APP mode)
- `src/core/config.py` (timestamp-based APP session ID)
- `src/core/auth.py` (skip PAT→TOKEN in APP mode)
- `src/agent/mlflow_ops.py` (add Volume sync functions)
- `src/agent/autonomous.py` (call sync at session boundaries)
