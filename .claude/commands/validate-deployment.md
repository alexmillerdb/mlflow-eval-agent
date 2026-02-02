---
description: Validate Databricks bundle deployment
argument-hint: "[--target dev|prod]"
---

Validate and test Databricks bundle deployment. Use this before shipping to production.

## Usage

```bash
# Validate dev deployment (default)
/validate-deployment

# Validate specific target
/validate-deployment --target prod
```

## Workflow

### Level 1: Bundle Validation
```bash
databricks bundle validate -t dev
```
**Success**: No validation errors

### Level 2: Deploy
```bash
databricks bundle deploy -t dev
```
**Success**: Bundle deploys without errors

### Level 3: Verify Deployment
```bash
databricks apps list
databricks apps get mlflow-eval-agent
```
**Success**: App appears in list with correct configuration

### Level 4: Job Run (Optional)
```bash
# Run the job to validate end-to-end
databricks bundle run eval_agent_job -t dev

# Check job status
databricks runs get-output --run-id <run-id>
```
**Success**: Job completes successfully

### Level 5: App Functional Testing
1. Get app URL from `databricks apps get` output
2. Open app in browser
3. Verify OAuth login works
4. Submit test query: "List the most recent 5 traces"
5. Verify streaming response appears
6. Check UC Volume for session files

## Success Criteria

| Level | Criteria |
|-------|----------|
| 1 | Bundle validates without errors |
| 2 | Bundle deploys successfully |
| 3 | App visible in `databricks apps list` |
| 4 | Job runs to completion (if executed) |
| 5 | Full UI functionality works |

## Common Issues

### Bundle Validation Fails

```bash
# Check for syntax errors in databricks.yml
databricks bundle validate -t dev 2>&1

# Common fixes:
# - Check variable references: ${var.name}
# - Verify paths exist
# - Check YAML indentation
```

### App Deployment Fails

```bash
# Check app.yaml configuration
cat app.yaml

# Verify required fields:
# - command is valid
# - env vars are configured
# - resources are reasonable
```

### OAuth Issues

```bash
# Verify app has OAuth enabled
databricks apps get mlflow-eval-agent

# Check app logs
databricks apps logs mlflow-eval-agent
```

### Job Run Fails

```bash
# Get detailed error
databricks runs get-output --run-id <run-id>

# Check cluster logs
databricks clusters list-logs <cluster-id>
```

## Rollback

If deployment fails in production:

```bash
# Redeploy previous version
git checkout <previous-tag>
databricks bundle deploy -t prod

# Or destroy and redeploy
databricks bundle destroy -t prod
databricks bundle deploy -t prod
```

## Example Run

```bash
/validate-deployment --target dev

# Output:
# ==========================================
# Deployment Validation - Target: dev
# ==========================================
#
# [1/5] Bundle Validation...
#   ✓ databricks.yml valid
#   ✓ app.yaml valid
#
# [2/5] Deploy...
#   ✓ Wheel built: mlflow_eval_agent-0.1.0-py3-none-any.whl
#   ✓ App deployed: mlflow-eval-agent
#   ✓ Job deployed: [dev] MLflow Eval Agent Job
#
# [3/5] Verify Deployment...
#   ✓ App: mlflow-eval-agent (RUNNING)
#   URL: https://workspace.databricks.com/apps/mlflow-eval-agent
#
# [4/5] Job Run (skipped - use --run-job to execute)
#
# [5/5] Manual Testing Required
#   → Open: https://workspace.databricks.com/apps/mlflow-eval-agent
#   → Test OAuth login
#   → Submit query: "List the most recent 5 traces"
#   → Verify streaming response
#
# ==========================================
# Deployment Validation: PASS
# ==========================================
```

## Production Checklist

Before deploying to production:

- [ ] All `/test-full-cycle` steps pass
- [ ] `/validate-deployment --target dev` passes
- [ ] Manual testing on dev complete
- [ ] Environment variables configured for prod
- [ ] UC Volume path correct for prod
- [ ] Model endpoint accessible from prod

```bash
# Final production deployment
/validate-deployment --target prod
```
