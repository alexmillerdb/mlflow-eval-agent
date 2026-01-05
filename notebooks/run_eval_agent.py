# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Evaluation Agent
# MAGIC
# MAGIC Autonomous agent for analyzing MLflow traces and building evaluation suites.
# MAGIC
# MAGIC **Usage:**
# MAGIC 1. Configure widgets in the sidebar (or pass as job parameters)
# MAGIC 2. Run all cells to execute the agent
# MAGIC 3. View results and traces in MLflow UI

# COMMAND ----------

# Widget Parameters
dbutils.widgets.text("experiment_id", "2181280362153689", "Target Experiment ID")
dbutils.widgets.text("volume_path", "/Volumes/users/alex_miller/mlflow-eval-agent", "Volume Path")
dbutils.widgets.text("max_iterations", "20", "Max Iterations")
dbutils.widgets.dropdown("model", "databricks-claude-opus-4-5",
    ["databricks-claude-opus-4-5", "databricks-claude-sonnet-4"], "Model")
dbutils.widgets.text("agent_experiment_id", "159502977489049", "Agent Traces Experiment")

# Authentication Configuration
dbutils.widgets.text("secret_scope", "mlflow-eval", "Secret Scope")
dbutils.widgets.text("secret_key", "databricks-token", "Secret Key")
dbutils.widgets.text("anthropic_api_key", "", "Anthropic API Key (optional)")
dbutils.widgets.text("disable_experimental_betas", "1", "Disable Experimental Betas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Package
# MAGIC
# MAGIC **Job runs**: Wheel is pre-installed via serverless environment (no action needed).
# MAGIC
# MAGIC **Interactive runs**: First copy wheel to Volume, then run the install cell.
# MAGIC ```
# MAGIC databricks fs cp dist/mlflow_eval_agent-*.whl dbfs:/Volumes/.../dist/
# MAGIC ```

# COMMAND ----------

# Check if package is installed, install from Volume if not
import subprocess
import sys

try:
    from src.agent import MLflowAgent
    print("Package already installed (job environment)")
except ImportError:
    print("Package not found, installing from Volume...")
    volume_path = dbutils.widgets.get("volume_path")
    wheel_path = f"{volume_path}/dist/mlflow_eval_agent-0.1.0-py3-none-any.whl"
    result = subprocess.run([sys.executable, "-m", "pip", "install", wheel_path, "-q"],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Install failed: {result.stderr}")
        raise RuntimeError(f"Failed to install from {wheel_path}")
    print(f"Installed from {wheel_path}")
    print("Note: You may need to restart the Python kernel for imports to work")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment Setup

# COMMAND ----------

# Set environment variables from widgets
import os
from databricks.sdk import WorkspaceClient

# Core configuration
os.environ["MLFLOW_EXPERIMENT_ID"] = dbutils.widgets.get("experiment_id")
os.environ["MLFLOW_AGENT_VOLUME_PATH"] = dbutils.widgets.get("volume_path")
os.environ["MLFLOW_AGENT_EXPERIMENT_ID"] = dbutils.widgets.get("agent_experiment_id")
os.environ["MODEL"] = dbutils.widgets.get("model")

# Derive Anthropic base URL from workspace
w = WorkspaceClient()
workspace_url = w.config.host.rstrip("/")
os.environ["ANTHROPIC_BASE_URL"] = f"{workspace_url}/serving-endpoints/anthropic"
os.environ["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = dbutils.widgets.get("disable_experimental_betas")

# Handle API key - prefer explicit key, fall back to secret scope
anthropic_api_key = dbutils.widgets.get("anthropic_api_key")
if anthropic_api_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
else:
    # Read from Databricks secret scope
    secret_scope = dbutils.widgets.get("secret_scope")
    secret_key = dbutils.widgets.get("secret_key")
    if secret_scope and secret_key:
        try:
            auth_token = dbutils.secrets.get(scope=secret_scope, key=secret_key)
            os.environ["ANTHROPIC_AUTH_TOKEN"] = auth_token
            print(f"Auth token loaded from secret scope: {secret_scope}/{secret_key}")
        except Exception as e:
            print(f"Warning: Could not read secret {secret_scope}/{secret_key}: {e}")

print(f"Target experiment: {os.environ['MLFLOW_EXPERIMENT_ID']}")
print(f"Agent traces experiment: {os.environ['MLFLOW_AGENT_EXPERIMENT_ID']}")
print(f"Volume path: {os.environ['MLFLOW_AGENT_VOLUME_PATH']}")
print(f"Model: {os.environ['MODEL']}")
print(f"Anthropic base URL: {os.environ['ANTHROPIC_BASE_URL']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize MLflow

# COMMAND ----------

# Import and initialize
from src.agent import setup_mlflow, run_autonomous
from src.config import Config

# Enable tracing + autolog (MLflow notebook integration shows traces automatically)
setup_mlflow()

# Load config to display settings
config = Config.from_env()

print(f"MLflow initialized")
print(f"  Analyzing experiment: {config.experiment_id}")
print(f"  Logging traces to: {config.agent_experiment_id}")
print(f"  Session storage: {os.environ.get('MLFLOW_AGENT_VOLUME_PATH', 'local')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Autonomous Evaluation
# MAGIC
# MAGIC The agent will:
# MAGIC 1. Analyze traces in the target experiment
# MAGIC 2. Identify patterns and issues
# MAGIC 3. Generate evaluation scorers and datasets
# MAGIC 4. Run validation and report results

# COMMAND ----------

# Run the autonomous evaluation loop
import asyncio

max_iters = int(dbutils.widgets.get("max_iterations"))
experiment_id = dbutils.widgets.get("experiment_id")

print(f"Starting autonomous evaluation (max {max_iters} iterations)...")
print("=" * 60)

# Handle async execution in notebook context
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Apply nest_asyncio for notebook compatibility
import nest_asyncio
nest_asyncio.apply()

# Run autonomous mode (standalone function, not a method)
result = loop.run_until_complete(
    run_autonomous(experiment_id, max_iterations=max_iters)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Results
# MAGIC
# MAGIC The autonomous evaluation has completed. Results are printed above.
# MAGIC
# MAGIC **View traces in MLflow UI:**
# MAGIC 1. Click "Experiments" in the sidebar
# MAGIC 2. Select the agent traces experiment (configured above)
# MAGIC 3. Click on traces to see detailed spans and tool calls
# MAGIC
# MAGIC **Output files** are saved to the Volume path configured in widgets.
