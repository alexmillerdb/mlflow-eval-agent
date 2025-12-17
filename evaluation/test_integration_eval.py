"""Integration tests using MLflow GenAI evaluation.

These tests run the full evaluation pipeline with real MLflow tracking.
They require:
- Databricks connection (DATABRICKS_HOST, DATABRICKS_TOKEN)
- MLflow tracking configured
- Experiment: /Users/alex.miller@databricks.com/mlflow-eval-agent-evaluations

Run with: pytest evaluation/test_integration_eval.py -v -m integration
"""

import pytest

from evaluation.test_data import (
    AGENT_ROUTING_CASES,
    ORDERING_CASES,
    TOOL_CALL_CASES,
    INTEGRATION_EVAL_DATA,
)
from evaluation.scorers import (
    agent_routing_accuracy,
    execution_order_scorer,
    tool_selection_accuracy,
    workspace_io_correctness,
)


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def mlflow_experiment(eval_experiment_path):
    """Set up MLflow experiment for integration tests."""
    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(eval_experiment_path)

    yield mlflow

    # Cleanup if needed


@pytest.fixture
def predict_fn():
    """Create predict function for evaluation.

    IMPORTANT: predict_fn receives **unpacked inputs as kwargs per GOTCHAS.md
    """

    def _predict(query: str, context: str = None):
        """Predict function that wraps the agent.

        Args:
            query: The user query (unpacked from inputs)
            context: Optional context (unpacked from inputs)

        Returns:
            dict with response and metadata
        """
        # Import locally for fast iteration (per patterns-evaluation.md Pattern 0)
        from src.agent import MLflowEvalAgent
        from src.config import EvalAgentConfig
        import asyncio

        try:
            config = EvalAgentConfig.from_env(validate=False)
            agent = MLflowEvalAgent(config)

            # Run the agent query
            async def run_query():
                results = []
                async for result in agent.query(query):
                    results.append(result)
                return results

            results = asyncio.run(run_query())

            # Extract response from results
            response_text = ""
            for r in results:
                if hasattr(r, "response") and r.response:
                    response_text += r.response

            return {
                "response": response_text or "No response generated",
                "result_count": len(results),
            }

        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "error": str(e),
            }

    return _predict


class TestAgentRoutingIntegration:
    """Integration tests for agent routing evaluation."""

    @pytest.mark.slow
    def test_agent_routing_evaluation(self, mlflow_experiment, predict_fn):
        """Run full agent routing evaluation with MLflow tracking."""
        from mlflow.genai.scorers import Guidelines

        # Use subset of cases for faster integration test
        eval_data = AGENT_ROUTING_CASES[:2]

        with mlflow_experiment.start_run(run_name="agent_routing_integration"):
            results = mlflow_experiment.genai.evaluate(
                data=eval_data,
                predict_fn=predict_fn,
                scorers=[
                    agent_routing_accuracy,
                    Guidelines(
                        name="routing_clarity",
                        guidelines="The response should clearly indicate which "
                        "sub-agent was selected for the task",
                    ),
                ],
            )

        assert results.run_id is not None
        assert "agent_routing_accuracy" in str(results.metrics) or len(results.metrics) > 0

        print(f"Run ID: {results.run_id}")
        print(f"Metrics: {results.metrics}")


class TestExecutionOrderIntegration:
    """Integration tests for execution order evaluation."""

    @pytest.mark.slow
    def test_execution_order_evaluation(self, mlflow_experiment, predict_fn):
        """Run execution order evaluation with MLflow tracking."""
        # Use first ordering case
        eval_data = ORDERING_CASES[:1]

        with mlflow_experiment.start_run(run_name="execution_order_integration"):
            results = mlflow_experiment.genai.evaluate(
                data=eval_data,
                predict_fn=predict_fn,
                scorers=[execution_order_scorer],
            )

        assert results.run_id is not None
        print(f"Run ID: {results.run_id}")
        print(f"Metrics: {results.metrics}")


class TestToolCallsIntegration:
    """Integration tests for tool call evaluation."""

    @pytest.mark.slow
    def test_tool_calls_evaluation(self, mlflow_experiment, predict_fn):
        """Run tool call evaluation with MLflow tracking."""
        # Use first tool call case
        eval_data = TOOL_CALL_CASES[:1]

        with mlflow_experiment.start_run(run_name="tool_calls_integration"):
            results = mlflow_experiment.genai.evaluate(
                data=eval_data,
                predict_fn=predict_fn,
                scorers=[
                    tool_selection_accuracy,
                    workspace_io_correctness,
                ],
            )

        assert results.run_id is not None
        print(f"Run ID: {results.run_id}")
        print(f"Metrics: {results.metrics}")


class TestCombinedEvaluation:
    """Combined evaluation with all scorers."""

    @pytest.mark.slow
    def test_full_evaluation_suite(self, mlflow_experiment, predict_fn):
        """Run comprehensive evaluation with all scorers."""
        from mlflow.genai.scorers import Guidelines, Safety

        with mlflow_experiment.start_run(run_name="full_evaluation_suite"):
            results = mlflow_experiment.genai.evaluate(
                data=INTEGRATION_EVAL_DATA,
                predict_fn=predict_fn,
                scorers=[
                    # Custom scorers
                    agent_routing_accuracy,
                    execution_order_scorer,
                    tool_selection_accuracy,
                    # Built-in scorers
                    Safety(),
                    Guidelines(
                        name="helpful",
                        guidelines="The response must be helpful and informative",
                    ),
                ],
            )

        assert results.run_id is not None
        print(f"\n=== Full Evaluation Results ===")
        print(f"Run ID: {results.run_id}")
        print(f"Metrics: {results.metrics}")

        # Log to MLflow for tracking
        for metric_name, value in results.metrics.items():
            print(f"  {metric_name}: {value}")


class TestPrecomputedOutputsEvaluation:
    """Test evaluation with pre-computed outputs (no predict_fn)."""

    def test_eval_with_precomputed_outputs(self, mlflow_experiment):
        """Evaluate pre-computed outputs without running agent."""
        from mlflow.genai.scorers import Guidelines

        # Pre-computed evaluation data
        precomputed_data = [
            {
                "inputs": {"query": "Analyze traces from experiment 123"},
                "outputs": {
                    "response": "I've analyzed the traces and found 5 error patterns..."
                },
                "expectations": {
                    "expected_agent": "trace_analyst",
                },
            },
            {
                "inputs": {"query": "Optimize the prompt context"},
                "outputs": {
                    "response": "Based on the analysis, here are my recommendations..."
                },
                "expectations": {
                    "expected_agent": "context_engineer",
                },
            },
        ]

        with mlflow_experiment.start_run(run_name="precomputed_outputs_eval"):
            # No predict_fn needed when outputs are pre-computed
            results = mlflow_experiment.genai.evaluate(
                data=precomputed_data,
                scorers=[
                    Guidelines(
                        name="response_quality",
                        guidelines="The response must be coherent and relevant to the query",
                    ),
                ],
            )

        assert results.run_id is not None
        print(f"Run ID: {results.run_id}")
        print(f"Metrics: {results.metrics}")


# Skip integration tests by default (require external services)
def pytest_configure(config):
    """Configure integration test markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
