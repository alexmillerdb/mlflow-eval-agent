"""Tests for EvalAgentConfig class."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.config import EvalAgentConfig


class TestEvalAgentConfigDefaults:
    """Test default values for EvalAgentConfig."""

    def test_default_databricks_fields(self):
        """Test Databricks-related default values."""
        config = EvalAgentConfig()
        assert config.databricks_host == ""
        assert config.databricks_token == ""
        assert config.databricks_config_profile == ""
        assert config.serverless_compute_id == "auto"
        assert config.cluster_id == ""

    def test_default_mlflow_fields(self):
        """Test MLflow-related default values."""
        config = EvalAgentConfig()
        assert config.experiment_id == ""
        assert config.experiment_path == "/Shared/mlflow-eval-agent"
        assert config.tracking_uri == "databricks"

    def test_default_model_fields(self):
        """Test model-related default values."""
        config = EvalAgentConfig()
        assert config.model == "sonnet"
        assert config.anthropic_base_url == ""
        assert config.anthropic_auth_token == ""

    def test_default_project_fields(self):
        """Test project-related default values."""
        config = EvalAgentConfig()
        assert config.project_name == "mlflow-eval-agent"
        assert config.environment == "development"

    def test_default_limit_fields(self):
        """Test limit-related default values."""
        config = EvalAgentConfig()
        assert config.max_turns == 50
        assert config.max_trace_results == 1000
        assert config.default_page_size == 50
        assert config.workspace_context_max_chars == 2000

    def test_default_path_fields(self):
        """Test path-related default values."""
        config = EvalAgentConfig()
        assert config.working_dir == Path.cwd()
        assert config.skills_dir.name == "skills"
        assert ".claude" in str(config.skills_dir)


class TestEvalAgentConfigPostInit:
    """Test __post_init__ behavior."""

    def test_sub_agents_model_inherits_from_model(self):
        """Test that sub_agents_model defaults to model value."""
        config = EvalAgentConfig(model="opus")
        assert config.sub_agents_model == "opus"

    def test_sub_agents_model_explicit_value(self):
        """Test that explicit sub_agents_model is preserved."""
        config = EvalAgentConfig(model="opus", sub_agents_model="haiku")
        assert config.sub_agents_model == "haiku"

    def test_sub_agents_model_empty_string_inherits(self):
        """Test that empty string sub_agents_model inherits from model."""
        config = EvalAgentConfig(model="sonnet", sub_agents_model="")
        assert config.sub_agents_model == "sonnet"


class TestEvalAgentConfigValidation:
    """Test validate() method."""

    def test_validate_raises_when_databricks_host_missing(self):
        """Test that validation fails when DATABRICKS_HOST is not set."""
        config = EvalAgentConfig(databricks_host="")
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "DATABRICKS_HOST is required" in str(exc_info.value)

    def test_validate_passes_when_databricks_host_set(self):
        """Test that validation passes when DATABRICKS_HOST is set."""
        config = EvalAgentConfig(databricks_host="https://example.cloud.databricks.com")
        config.validate()  # Should not raise

    def test_validate_passes_without_databricks_token(self):
        """Test that validation passes without DATABRICKS_TOKEN (profile auth)."""
        config = EvalAgentConfig(
            databricks_host="https://example.cloud.databricks.com",
            databricks_token="",
        )
        config.validate()  # Should not raise


class TestEvalAgentConfigFromEnv:
    """Test from_env() class method."""

    def test_from_env_loads_databricks_host(self):
        """Test that from_env loads DATABRICKS_HOST."""
        with mock.patch.dict(os.environ, {"DATABRICKS_HOST": "https://test.databricks.com"}):
            config = EvalAgentConfig.from_env()
        assert config.databricks_host == "https://test.databricks.com"

    def test_from_env_loads_all_databricks_fields(self):
        """Test that from_env loads all Databricks-related fields."""
        env_vars = {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "dapi123",
            "DATABRICKS_CONFIG_PROFILE": "my-profile",
            "DATABRICKS_SERVERLESS_COMPUTE_ID": "compute-123",
            "DATABRICKS_CLUSTER_ID": "cluster-456",
        }
        with mock.patch.dict(os.environ, env_vars, clear=False):
            config = EvalAgentConfig.from_env()
        assert config.databricks_host == "https://test.databricks.com"
        assert config.databricks_token == "dapi123"
        assert config.databricks_config_profile == "my-profile"
        assert config.serverless_compute_id == "compute-123"
        assert config.cluster_id == "cluster-456"

    def test_from_env_loads_mlflow_fields(self):
        """Test that from_env loads MLflow-related fields."""
        env_vars = {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "MLFLOW_EXPERIMENT_ID": "exp-123",
            "MLFLOW_TRACKING_URI": "https://mlflow.example.com",
        }
        with mock.patch.dict(os.environ, env_vars, clear=False):
            config = EvalAgentConfig.from_env()
        assert config.experiment_id == "exp-123"
        assert config.tracking_uri == "https://mlflow.example.com"

    def test_from_env_loads_model_fields(self):
        """Test that from_env loads model-related fields."""
        env_vars = {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DABS_MODEL": "databricks-claude-opus",
            "SUB_AGENTS_MODEL": "databricks-claude-sonnet",
            "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
            "ANTHROPIC_AUTH_TOKEN": "auth-token-123",
        }
        with mock.patch.dict(os.environ, env_vars, clear=False):
            config = EvalAgentConfig.from_env()
        assert config.model == "databricks-claude-opus"
        assert config.sub_agents_model == "databricks-claude-sonnet"
        assert config.anthropic_base_url == "https://api.anthropic.com"
        assert config.anthropic_auth_token == "auth-token-123"

    def test_from_env_loads_project_fields(self):
        """Test that from_env loads project-related fields."""
        env_vars = {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "PROJECT_NAME": "my-project",
            "ENVIRONMENT": "production",
        }
        with mock.patch.dict(os.environ, env_vars, clear=False):
            config = EvalAgentConfig.from_env()
        assert config.project_name == "my-project"
        assert config.environment == "production"

    def test_from_env_uses_defaults_for_missing_vars(self):
        """Test that from_env uses defaults when env vars are not set."""
        # Use a non-existent .env file to prevent loading from project .env
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_env = os.path.join(tmpdir, ".env")
            # Create empty .env file
            Path(fake_env).write_text("DATABRICKS_HOST=https://test.databricks.com\n")
            # Clear env vars to test defaults
            env_clear = {
                "DATABRICKS_HOST": "",
                "DATABRICKS_SERVERLESS_COMPUTE_ID": "",
                "MLFLOW_TRACKING_URI": "",
                "DABS_MODEL": "",
                "PROJECT_NAME": "",
                "ENVIRONMENT": "",
            }
            with mock.patch.dict(os.environ, {}, clear=True):
                config = EvalAgentConfig.from_env(env_file=fake_env)
        assert config.serverless_compute_id == "auto"
        assert config.tracking_uri == "databricks"
        assert config.model == "sonnet"
        assert config.project_name == "mlflow-eval-agent"
        assert config.environment == "development"

    def test_from_env_validates_by_default(self):
        """Test that from_env validates by default."""
        # Use a non-existent .env file and clear env to test validation
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_env = os.path.join(tmpdir, ".env")
            Path(empty_env).write_text("")  # Empty .env file
            with mock.patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError) as exc_info:
                    EvalAgentConfig.from_env(env_file=empty_env)
        assert "DATABRICKS_HOST is required" in str(exc_info.value)

    def test_from_env_skip_validation(self):
        """Test that from_env can skip validation."""
        # Use a non-existent .env file and clear env to test skip validation
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_env = os.path.join(tmpdir, ".env")
            Path(empty_env).write_text("")  # Empty .env file
            with mock.patch.dict(os.environ, {}, clear=True):
                config = EvalAgentConfig.from_env(env_file=empty_env, validate=False)
        assert config.databricks_host == ""


class TestEvalAgentConfigFromEnvFile:
    """Test from_env() with .env file loading."""

    def test_from_env_loads_dotenv_file(self):
        """Test that from_env loads from a .env file."""
        env_content = """
DATABRICKS_HOST=https://dotenv-test.databricks.com
DATABRICKS_TOKEN=dapi-from-file
DABS_MODEL=opus
PROJECT_NAME=dotenv-project
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(env_content)
            env_file = f.name

        try:
            # Clear relevant env vars to ensure we're loading from file
            env_clear = {
                "DATABRICKS_HOST": "",
                "DATABRICKS_TOKEN": "",
                "DABS_MODEL": "",
                "PROJECT_NAME": "",
            }
            with mock.patch.dict(os.environ, env_clear, clear=False):
                # Remove keys entirely for test
                for key in env_clear:
                    os.environ.pop(key, None)
                config = EvalAgentConfig.from_env(env_file=env_file)

            assert config.databricks_host == "https://dotenv-test.databricks.com"
            assert config.databricks_token == "dapi-from-file"
            assert config.model == "opus"
            assert config.project_name == "dotenv-project"
        finally:
            os.unlink(env_file)

    def test_from_env_existing_env_vars_take_precedence(self):
        """Test that existing env vars are not overwritten by .env file."""
        env_content = """
DATABRICKS_HOST=https://from-file.databricks.com
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(env_content)
            env_file = f.name

        try:
            # Set env var that should take precedence
            with mock.patch.dict(
                os.environ,
                {"DATABRICKS_HOST": "https://from-env-var.databricks.com"},
                clear=False,
            ):
                config = EvalAgentConfig.from_env(env_file=env_file)

            # Existing env var should take precedence over .env file
            assert config.databricks_host == "https://from-env-var.databricks.com"
        finally:
            os.unlink(env_file)


class TestEvalAgentConfigSubAgentsModelInheritance:
    """Test sub_agents_model inheritance with from_env()."""

    def test_sub_agents_model_inherits_from_dabs_model_in_from_env(self):
        """Test that SUB_AGENTS_MODEL inherits from DABS_MODEL when not set."""
        # Use isolated .env file to prevent loading from project .env
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = os.path.join(tmpdir, ".env")
            Path(env_file).write_text(
                "DATABRICKS_HOST=https://test.databricks.com\n"
                "DABS_MODEL=databricks-claude-opus\n"
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                config = EvalAgentConfig.from_env(env_file=env_file)

        assert config.model == "databricks-claude-opus"
        assert config.sub_agents_model == "databricks-claude-opus"

    def test_sub_agents_model_explicit_in_from_env(self):
        """Test that explicit SUB_AGENTS_MODEL is preserved in from_env."""
        # Use isolated .env file to prevent loading from project .env
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = os.path.join(tmpdir, ".env")
            Path(env_file).write_text(
                "DATABRICKS_HOST=https://test.databricks.com\n"
                "DABS_MODEL=databricks-claude-opus\n"
                "SUB_AGENTS_MODEL=databricks-claude-haiku\n"
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                config = EvalAgentConfig.from_env(env_file=env_file)

        assert config.model == "databricks-claude-opus"
        assert config.sub_agents_model == "databricks-claude-haiku"
