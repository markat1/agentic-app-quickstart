import asyncio
from dotenv import load_dotenv

import os
from agentic_app_quickstart.week_1.solution.telemetry import init_tracing
from agentic_app_quickstart.week_2.solution.monitoring.evaluators import (
    get_spans_from_phoenix,
    create_test_dataset,
    evaluate_hallucination,
)
from agentic_app_quickstart.week_2.solution.monitoring.summary import summarize_results
from agentic_app_quickstart.week_2.solution.monitoring.metrics import start_span


async def main():
    load_dotenv()
    
    init_tracing(
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
    )

    spans_dataset = get_spans_from_phoenix(project_name="agentic_app_quickstart", sample=10)
    test_dataset = create_test_dataset()

    if spans_dataset:
        with start_span("evaluation.hallucination", {"dataset.name": "phoenix_spans", "dataset.size": len(spans_dataset)}):
            spans_results = evaluate_hallucination(spans_dataset)
            summarize_results("Phoenix spans", spans_results, spans_dataset)
    if test_dataset:
        with start_span("evaluation.hallucination", {"dataset.name": "test_dataset", "dataset.size": len(test_dataset)}):
            test_results = evaluate_hallucination(test_dataset)
            summarize_results("Test dataset", test_results, test_dataset)


if __name__ == "__main__":
    asyncio.run(main())


