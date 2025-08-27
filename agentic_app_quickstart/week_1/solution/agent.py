from __future__ import annotations

import os
from agents import Agent, ModelSettings

from agentic_app_quickstart.examples.helpers import get_model

from agentic_app_quickstart.week_1.solution.telemetry import init_tracing
from openinference.semconv.trace import SpanAttributes

from .tools import (
    get_column_names,
    calculate_column_average,
    count_rows_with_value,
    calculate_filtered_column_average,
    create_histogram,
    create_scatter_plot,
    create_bar_chart,
    find_column_maximum,
    find_column_minimum,
    calculate_percentage,
    calculate_ratio,
    calculate_column_percentile,
    find_outlier_limits,
    remove_outliers,
    find_correlation,
    get_correlation_matrix,
    compare_with_dataset,
    compare_csv_files
)


__all__ = [
    "create_analysis_agent",
    "create_data_loader_agent",
    "create_communication_agent",
]

tracer, _ = init_tracing()



@tracer.start_as_current_span(name="analysis_agent", attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: "agent"})
def create_analysis_agent() -> Agent:
    """
    Agent that receives a pandas DataFrame (in context) and uses tools to analyze it.
    """
    return Agent(
        name="analysis_agent",
        model=get_model(),
        model_settings=ModelSettings(temperature=0.1),
        instructions=(
            "You are a data analysis assistant. The conversation context contains a "
            "pandas DataFrame. Use the available tools to inspect columns and compute "
            "results. Prefer numeric columns for aggregations. If the question cannot "
            "be answered with the provided data, state that clearly. Return a concise "
            "answer and include the key numbers used."
            "Plan your steps and then execute them."
            "First use the get_file function to load the file into memory and then use the available tools to analyze the data."
        ),
        tools=[
            get_column_names,
            calculate_column_average,
            count_rows_with_value,
            calculate_filtered_column_average,
            create_histogram,
            create_scatter_plot,
            create_bar_chart,
            find_column_maximum,
            find_column_minimum,
            calculate_percentage,
            calculate_ratio,
            calculate_column_percentile,
            find_outlier_limits,
            remove_outliers,
            find_correlation,
            get_correlation_matrix,
            compare_with_dataset,
            compare_csv_files,
        ],
    )


@tracer.start_as_current_span(name="data_loader_agent", attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: "agent"})
def create_data_loader_agent() -> Agent:
    """
    Agent that decides which dataset best answers the user's question.
    Must return exactly one of: employees, weather, sales (lowercase).
    """
    return Agent(
        name="data_loader_agent",
        model=get_model(),
        model_settings=ModelSettings(temperature=0.0),
        instructions=(
            "Choose the dataset that best answers the user's question. The available "
            "datasets are:\n"
            "- employees: HR-style employee records\n"
            "- weather: daily weather observations\n"
            "- sales: transactional sales data\n\n"
            "Return ONLY a single lowercase word: employees, weather, or sales."
            "The list below contains the file paths and names for different topics"
            """
            "employees": "data/employee_data.csv"
            "weather": "data/weather_data.csv"
            "sales": "data/sample_sales.csv"
            """
            "Return the path to the csv file"
        ),
    )

@tracer.start_as_current_span(name="communication_agent", attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: "agent"})
def create_communication_agent() -> Agent:
    """
    Agent that turns the analysis output into a clear, user-friendly final answer.
    The context passed to this agent will be the analysis result text.
    """
    return Agent(
        name="communication_agent",
        model=get_model(),
        model_settings=ModelSettings(temperature=0.3),
        instructions=(
            "You are a communication specialist. You will receive the user's question "
            "and the analysis agent: "
            "and the analysis result as context. Provide a short, friendly final "
            "answer in plain language. Do not reveal tool calls or intermediate steps. "
            "If the analysis indicates the question cannot be answered, explain why."
        ),
    )
