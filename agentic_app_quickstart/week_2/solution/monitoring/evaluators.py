import os
from typing import Iterable, Sequence

import opentelemetry.trace
from pandas import DataFrame
from phoenix import Client
from phoenix.evals import OpenAIModel, llm_classify
from agentic_app_quickstart.examples.helpers import get_model
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace as trace_api
from agentic_app_quickstart.week_1.solution.telemetry import init_tracing
from openinference.instrumentation.openai import OpenAIInstrumentor

tracer, tracer_provider = init_tracing()

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

def get_spans_from_phoenix(
    project_name: str,
    *,
    sample: int | None = 50,
    include_agents: Sequence[str] | None = ("DataLoaderAgent", "AnalysisAssistantAgent", "CommunicationAgent"),
) -> list[dict]:
    endpoint = os.getenv("PHOENIX_ENDPOINT")
    client = Client(endpoint=endpoint, api_key=os.getenv("PHOENIX_API_KEY"))
    
    try:
        spans_from_phoenix : DataFrame = client.get_spans_dataframe(project_name=project_name, limit=500)
        if include_agents and "attributes.agent.name" in spans_from_phoenix.columns:
            spans_from_phoenix = spans_from_phoenix[spans_from_phoenix["attributes.agent.name"].isin(include_agents)]
        if sample:
            spans_from_phoenix = spans_from_phoenix.sample(n=min(sample, len(spans_from_phoenix)))
    except Exception:
        return []
    
    # Convert to list of dicts
    return [
        {
            "input": row.get("attributes.input.value", ""),
            "output": row.get("attributes.output.value", ""),
            "agent_name": row.get("attributes.agent.name", ""),
        }
        for _, row in spans_from_phoenix.iterrows()
    ]

def classify_user_intent(
        user_prompt: str,
        tracer: opentelemetry.trace.Tracer):
    with tracer.start_as_current_span("classify_user_intent") as span:
        span.set_attribute("mycompany.user_prompt", user_prompt)
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(SpanAttributes.LLM_COST_COMPLETION_TOKENS, 10)
        
        span.set_status(trace_api.StatusCode.OK)


HALLUCINATION_PROMPT_TEMPLATE = (
    "You are auditing outputs for hallucinations with awareness of the producing agent's role.\n"
    "Classify each case as one of: grounded, hallucination.\n"
    "\n"
    "Agent: {agent_name}\n"
    "User input:\n{input}\n\n"
    "Assistant output:\n{output}\n\n"
    "Role-specific guidance:\n"
    "- If Agent is DataLoaderAgent: The choice must match the domain implied by the user input (e.g., employees vs weather vs sales).\n"
    "  Label hallucination if the dataset suggestion is unrelated to the user input.\n"
    "- If Agent is AnalysisAssistantAgent: Calculations and statistics must be derivable from the given input/context.\n"
    "  Label hallucination if numbers or findings are introduced without grounding.\n"
    "- If Agent is CommunicationAgent: The response must faithfully rephrase the underlying analysis without changing facts.\n"
    "  Label hallucination if it alters values or adds unsupported details.\n"
    "\n"
    "Definitions:\n"
    "- grounded: Supported by or logically follows from the user input/context.\n"
    "- hallucination: Introduces facts/details unsupported by the user input/context.\n"
    "\n"
    "Return only one label: grounded or hallucination."
)


def evaluate_hallucination(dataset: Iterable[dict]):
    rails = ["grounded", "hallucination"]
    return llm_classify(
        data=list(dataset),
        template=HALLUCINATION_PROMPT_TEMPLATE,
        model=get_model(),
        rails=rails,
        provide_explanation=True,
        concurrency=5,
    )


def create_test_dataset() -> list[dict]:
    """
    Small hand-crafted dataset covering the three Week 1 agents.

    Each item contains:
    - agent_name: one of DataLoaderAgent | AnalysisAssistantAgent | CommunicationAgent
    - input: the user's input (or upstream agent message)
    - output: the agent's output to evaluate
    """
    return [
        # -----------------
        # DataLoaderAgent
        # -----------------
        # grounded
        {
            "agent_name": "DataLoaderAgent",
            "input": "What's the average employee salary?",
            "output": "employees",
        },
        {
            "agent_name": "DataLoaderAgent",
            "input": "Show me the rainfall stats for last week.",
            "output": "weather",
        },
        {
            "agent_name": "DataLoaderAgent",
            "input": "How many orders did we have last quarter?",
            "output": "sales",
        },
        # hallucination (wrong domain)
        {
            "agent_name": "DataLoaderAgent",
            "input": "How hot will it be today?",
            "output": "sales",
        },
        {
            "agent_name": "DataLoaderAgent",
            "input": "What was the total monthly revenue?",
            "output": "weather",
        },

        # -----------------
        # AnalysisAssistantAgent
        # -----------------
        # grounded
        {
            "agent_name": "AnalysisAssistantAgent",
            "input": "On the employees dataset, compute the median salary.",
            "output": "The median salary is 71,000.",
        },
        {
            "agent_name": "AnalysisAssistantAgent",
            "input": "From the weather dataset, what was the minimum temperature last week?",
            "output": "The minimum temperature was -2°C.",
        },
        {
            "agent_name": "AnalysisAssistantAgent",
            "input": "In the sales dataset, what is the correlation between price and quantity?",
            "output": "The correlation between price and quantity is -0.32.",
        },
        # hallucination (adds unsupported or off-topic detail)
        {
            "agent_name": "AnalysisAssistantAgent",
            "input": "List numeric columns available in the employees dataset.",
            "output": "Numeric columns: salary, bonus, equity (equity not present).",
        },
        {
            "agent_name": "AnalysisAssistantAgent",
            "input": "From the weather dataset, compute the average rainfall.",
            "output": "Average rainfall was 1,200 inches due to a hurricane last night.",
        },

        # -----------------
        # CommunicationAgent
        # -----------------
        # grounded (faithful rephrase)
        {
            "agent_name": "CommunicationAgent",
            "input": "User question: What's the average salary?\nRaw answer: 73466.67",
            "output": "The average salary is 73,466.67.",
        },
        {
            "agent_name": "CommunicationAgent",
            "input": "User question: What is the average rainfall?\nRaw answer: 12.5",
            "output": "The average rainfall was 12.5 mm.",
        },
        # hallucination (changes value or adds unsupported detail)
        {
            "agent_name": "CommunicationAgent",
            "input": "User question: What's the average salary?\nRaw answer: 73466.67",
            "output": "The average salary is 80,000.",
        },
        {
            "agent_name": "CommunicationAgent",
            "input": "User question: What is the average rainfall?\nRaw answer: 12.5",
            "output": "In Q4, the average rainfall was 60 mm across all regions.",
        },
    ]


