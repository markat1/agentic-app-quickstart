import asyncio
from dotenv import load_dotenv

import os
from phoenix.otel import register
from agentic_app_quickstart.week_2.solution.monitoring.metrics import (
    start_reasoning_span,
    start_tool_span,
    start_llm_span,
)
from agentic_app_quickstart.week_2.solution.monitoring.evaluators import (
    create_phoenix_spans_dataset,
    create_test_dataset,
    evaluate_hallucination,
)
from agentic_app_quickstart.week_2.solution.monitoring.summary import summarize_results


async def main():
    load_dotenv()
    
    # Initialize tracing
    endpoint = os.getenv("PHOENIX_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "PHOENIX_ENDPOINT is not set. Define it in your environment or .env file."
        )
    
    register(
        endpoint=endpoint,
        project_name="agentic_app_quickstart",
        protocol="http/protobuf",
        auto_instrument=True,
    )

    with start_reasoning_span(user_query="example question", agent_name="Week2Agent"):
        with start_tool_span(tool_name="csv_loader", tool_input="employees.csv"):
            pass
        with start_llm_span(model="gpt-4.1", prompt_tokens=42, completion_tokens=84, total_cost=0.0012):
            pass

    # Evaluation runs: Phoenix spans and a small built-in test dataset
    spans_dataset = create_phoenix_spans_dataset(project_name="agentic_app_quickstart", sample=10)
    test_dataset = create_test_dataset()

    if spans_dataset:
        spans_results = evaluate_hallucination(spans_dataset)
        summarize_results("Phoenix spans", spans_results, spans_dataset)
    if test_dataset:
        test_results = evaluate_hallucination(test_dataset)
        summarize_results("Test dataset", test_results, test_dataset)


if __name__ == "__main__":
    asyncio.run(main())


