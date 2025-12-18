"""
Evaluation dataset for Multi-Genie Orchestrator agent.

Built from production traces with pre-computed outputs.
Strategy: traces (no predict_fn needed - evaluating existing responses)

Dataset covers:
- Sales pipeline queries (customer_sales routing)
- Supply chain queries (supply_chain routing)
- System info queries (no Genie needed)
- Error cases for robustness testing
- Suboptimal routing cases for quality analysis
"""

# Evaluation dataset with pre-computed inputs and outputs from traces
# Format follows Pattern 4 from patterns-datasets.md
EVAL_DATASET = [
    # Case 1: System info query - supervisor handles directly (no Genie calls)
    # Trace: tr-a42854090ea8d5dbed6d596c3f69a06a
    # This tests correct handling of meta-questions about the system itself
    {
        "inputs": {
            "messages": [{"role": "user", "content": "What does the agent do?"}]
        },
        "outputs": {
            "response": """This agent system orchestrates specialized Genies to answer business questions.

The agent works as a supervisor that:
1. Routes questions to specialized Genies with access to specific data domains
2. Currently has access to two specialized Genies:
   - Customer Sales Genie: Provides information about the sales pipeline
   - Supply Chain Genie: Provides information about supply chain optimization tables

When you ask a question, the system determines which Genie(s) can best answer it, routes your question appropriately, and returns the relevant information from the specialized data sources.""",
            "workflow_summary": """# Multi-Agent Workflow Summary

## User Query
The user asked "What does the agent do?" - seeking to understand the purpose and functionality of this agent system.

## Agents Used
No specialized Genie agents were invoked for this query, as it was a general question about the system itself rather than a domain-specific business question requiring specialized data access.

## Workflow Execution
The supervisor agent directly handled this query without routing to specialized Genies since it was asking about the system's own functionality.

## Key Insights
The system functions as an orchestration layer that:
- Routes business questions to specialized Genie agents
- Currently supports two specialized Genies:
  - Customer Sales Genie (sales pipeline data)
  - Supply Chain Genie (supply chain optimization data)
- Determines which Genie(s) can best answer each question
- Returns relevant information from specialized data sources

## Final Outcome
The user received a comprehensive explanation of the agent system's purpose, capabilities, and workflow, providing clarity on how to effectively use the system for business inquiries requiring specialized data access.""",
            "agents_used": []
        },
        "metadata": {
            "trace_id": "tr-a42854090ea8d5dbed6d596c3f69a06a",
            "query_type": "system_info",
            "expected_routing": "none",
            "status": "OK",
            "latency_ms": 13826
        }
    },

    # Case 2: Sales pipeline query - suboptimal routing (called both Genies)
    # Trace: tr-d45af7104661af36e7ccd1e81a76f2d8
    # This tests routing efficiency - supply_chain was called unnecessarily
    {
        "inputs": {
            "messages": [{"role": "user", "content": "How's my pipeline just in the americas and by segment?"}]
        },
        "outputs": {
            "response": """Sales Pipeline for the Americas by Segment:

|    | company_size_segment__c   |   sum(pipeline_amount) |
|---:|:--------------------------|-----------------------:|
|  0 | ENT                       |                15999.6 |
|  1 | MM                        |                18003.1 |
|  2 | SMB                       |                26001.2 |""",
            "workflow_summary": """# Multi-Agent Workflow Summary

## Query Analysis
The user requested information about their sales pipeline specifically in the Americas region, broken down by segment.

## Workflow Execution
Two specialized agents were consulted to address this query:

1. **Customer Sales Agent** (Primary)
   - Selected because the query directly related to sales pipeline data
   - Successfully retrieved and analyzed sales pipeline figures for the Americas region
   - Provided a breakdown by company size segments (ENT, MM, SMB)

2. **Supply Chain Agent** (Secondary)
   - Consulted to check for any complementary supply chain data
   - Determined that their available data could not be filtered by geographic regions or segments
   - No additional supply chain insights could be provided for this specific query

## Key Findings
The Customer Sales Agent identified that in the Americas region:
- Small and Medium Business (SMB) segment has the highest pipeline value at $26,001.20
- Mid-Market (MM) segment shows a pipeline value of $18,003.10
- Enterprise (ENT) segment has a pipeline value of $15,999.60

## Conclusion
The analysis reveals that the SMB segment currently represents the strongest sales pipeline opportunity in the Americas region, accounting for approximately 43% of the total regional pipeline value. The MM and ENT segments represent approximately 30% and 27% of the pipeline respectively.

This segment-based view provides valuable insight for regional sales strategy and resource allocation decisions.""",
            "agents_used": ["customer_sales", "supply_chain"]
        },
        "metadata": {
            "trace_id": "tr-d45af7104661af36e7ccd1e81a76f2d8",
            "query_type": "sales_pipeline",
            "expected_routing": "customer_sales",  # supply_chain was called unnecessarily
            "actual_routing": ["customer_sales", "supply_chain"],
            "routing_quality": "suboptimal",
            "status": "OK",
            "latency_ms": 39822
        }
    },

    # Case 3: Supply chain query - correct routing (supply_chain only)
    # Trace: tr-a077529e0ca08f9cc1692bf7e571fa82
    # This tests correct routing for supply chain domain questions
    {
        "inputs": {
            "messages": [{"role": "user", "content": "What are our top 3 forecasted raw material shortages?"}]
        },
        "outputs": {
            "response": """Top 3 forecasted raw material shortages:

|    | raw   |   total_demand |   total_supply |   shortage |
|---:|:------|---------------:|---------------:|-----------:|
|  0 | T26FB |          53226 |          32435 |      20791 |
|  1 | RLNGY |          50436 |          29695 |      20741 |
|  2 | 568HR |          93609 |          75610 |      17999 |""",
            "workflow_summary": """# Multi-Agent Workflow Summary

## Query Overview
The user requested information about the top 3 forecasted raw material shortages.

## Workflow Execution
The system identified this as a supply chain query and appropriately routed it to the **supply_chain** agent. No other agents were needed as this specialized agent had all the necessary data to address the specific question about material shortages.

## Key Findings
The supply_chain agent identified the following critical shortages:

| Material | Total Demand | Total Supply | Shortage |
|:---------|-------------:|-------------:|---------:|
| T26FB    | 53,226       | 32,435       | 20,791   |
| RLNGY    | 50,436       | 29,695       | 20,741   |
| 568HR    | 93,609       | 75,610       | 17,999   |

## Outcome
The analysis revealed three raw materials with significant forecasted shortages. Materials T26FB and RLNGY show nearly identical shortage levels (approximately 20,700 units each), while 568HR has a slightly lower but still substantial shortage of nearly 18,000 units. These findings highlight potential supply chain vulnerabilities that may require immediate attention for production planning and procurement strategies.""",
            "agents_used": ["supply_chain"]
        },
        "metadata": {
            "trace_id": "tr-a077529e0ca08f9cc1692bf7e571fa82",
            "query_type": "supply_chain",
            "expected_routing": "supply_chain",
            "actual_routing": ["supply_chain"],
            "routing_quality": "optimal",
            "status": "OK",
            "latency_ms": 28496
        }
    },

    # Case 4: Error case - rate limit exceeded
    # Trace: tr-27590fb91da88b6e7db29b35c60e13ee
    # This tests error handling and robustness
    {
        "inputs": {
            "messages": [{"role": "user", "content": "What are our top 3 forecasted raw material shortages?"}]
        },
        "outputs": {
            "response": None,  # No response due to error
            "workflow_summary": None,
            "agents_used": ["supply_chain"],
            "error": "RateLimitError: Error code: 429 - REQUEST_LIMIT_EXCEEDED: Exceeded workspace output tokens per minute rate limit for databricks-claude-3-7-sonnet."
        },
        "metadata": {
            "trace_id": "tr-27590fb91da88b6e7db29b35c60e13ee",
            "query_type": "supply_chain",
            "expected_routing": "supply_chain",
            "status": "ERROR",
            "error_type": "rate_limit",
            "latency_ms": 10628
        }
    },

    # Case 5: Synthetic - Multi-domain query (would need both Genies)
    # Based on observed patterns, this type of query would be correctly routed to both
    {
        "inputs": {
            "messages": [{"role": "user", "content": "Show me our sales pipeline and any related supply chain risks"}]
        },
        "outputs": None,  # Synthetic case - no pre-computed output
        "metadata": {
            "trace_id": None,
            "query_type": "multi_domain",
            "expected_routing": ["customer_sales", "supply_chain"],
            "status": "synthetic",
            "notes": "Synthetic case for testing multi-domain routing - both Genies should be called"
        }
    },

    # Case 6: Edge case - Ambiguous query
    # Testing how the supervisor handles queries that could go either way
    {
        "inputs": {
            "messages": [{"role": "user", "content": "What's happening with our pipeline?"}]
        },
        "outputs": None,  # Synthetic case
        "metadata": {
            "trace_id": None,
            "query_type": "ambiguous",
            "expected_routing": "customer_sales",  # "pipeline" typically refers to sales pipeline
            "status": "synthetic",
            "notes": "Ambiguous query - 'pipeline' could mean sales or supply chain. Expected: sales"
        }
    },
]

# Filtered dataset for evaluation (only cases with pre-computed outputs)
EVAL_DATASET_WITH_OUTPUTS = [
    case for case in EVAL_DATASET
    if case.get("outputs") is not None and case["outputs"].get("response") is not None
]

# Dataset for routing quality evaluation
ROUTING_EVAL_DATASET = [
    case for case in EVAL_DATASET
    if case["metadata"].get("expected_routing") is not None
]

# Dataset for error case testing
ERROR_CASES = [
    case for case in EVAL_DATASET
    if case["metadata"].get("status") == "ERROR"
]

# Successful cases only
SUCCESS_CASES = [
    case for case in EVAL_DATASET
    if case["metadata"].get("status") == "OK"
]


def get_eval_data_for_mlflow():
    """
    Returns evaluation data in MLflow-compatible format.
    Only includes cases with pre-computed outputs (no predict_fn needed).

    Format: List of dicts with 'inputs' and 'outputs' keys
    """
    return [
        {
            "inputs": case["inputs"],
            "outputs": {
                "response": case["outputs"]["response"],
                "workflow_summary": case["outputs"].get("workflow_summary", "")
            }
        }
        for case in EVAL_DATASET_WITH_OUTPUTS
    ]


def get_eval_data_with_metadata():
    """
    Returns evaluation data with metadata for detailed analysis.
    Includes trace_id, query_type, routing info, etc.
    """
    return [
        {
            "inputs": case["inputs"],
            "outputs": case.get("outputs"),
            "metadata": case["metadata"]
        }
        for case in EVAL_DATASET
        if case.get("outputs") is not None
    ]


if __name__ == "__main__":
    print(f"Total cases: {len(EVAL_DATASET)}")
    print(f"Cases with outputs: {len(EVAL_DATASET_WITH_OUTPUTS)}")
    print(f"Success cases: {len(SUCCESS_CASES)}")
    print(f"Error cases: {len(ERROR_CASES)}")
    print(f"Routing eval cases: {len(ROUTING_EVAL_DATASET)}")

    print("\nQuery types:")
    for case in EVAL_DATASET:
        print(f"  - {case['metadata']['query_type']}: {case['inputs']['messages'][0]['content'][:50]}...")
