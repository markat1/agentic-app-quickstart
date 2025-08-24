import asyncio
import os
import pandas as pd
from dotenv import load_dotenv

from agents import Runner, SQLiteSession
from agentic_app_quickstart.week_1.solution.agent import (
    create_analysis_agent,
    create_data_loader_agent,
    create_communication_agent,
)
from agentic_app_quickstart.week_1.solution.hooks import FollowUpHooks, prompt_next_question
from agentic_app_quickstart.week_1.solution.telemetry import init_tracing


async def main():
    """
    Demonstrates function calling across three agents:
      1) Load CSV data
      2) Ask the user a question
      3) Analyze and communicate the answer
    """
    # Env and tracing
    load_dotenv()
    project = os.environ.get("PHOENIX_PROJECT_NAME")
    init_tracing(
        project_name=project,
        auto_instrument=True,
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
    )

    session = SQLiteSession(session_id=123)
    hooks = FollowUpHooks()
   
    print("You can ask questions about the data. Type 'exit' to quit.\n")

    # Main Q&A loop
    while True:
        # Get question (with hook suggestions)
        question = prompt_next_question(hooks)
        if question is None:
            print("Goodbye!")
            break

        # Decide which dataset to load based on the question
        loader_result = await Runner.run(
            starting_agent=create_data_loader_agent(),
            input=question,
            session=session,
        )
        key = str(getattr(loader_result, "final_output", "")).strip().lower()

        path_map = {
            "employees": "data/employee_data.csv",
            "weather": "data/weather_data.csv",
            "sales": "data/sample_sales.csv",
        }

        # Auto-select the best dataset based on AI recommendation
        if key in path_map:
            csv_path = path_map[key]
            print(f"Data loader selected: {key} -> {csv_path}")
        else:
            # Fallback: try to infer from question content
            question_lower = question.lower()
            if any(word in question_lower for word in ["employee", "salary", "performance", "hr"]):
                key = "employees"
            elif any(word in question_lower for word in ["weather", "temperature", "humidity", "climate"]):
                key = "weather"
            elif any(word in question_lower for word in ["sales", "price", "quantity", "transaction"]):
                key = "sales"
            else:
                key = "employees"  # Default fallback
            csv_path = path_map[key]
            print(f"Auto-selected dataset: {key} -> {csv_path}")

        # Load CSV into context
        content = pd.read_csv(csv_path)
       
        # Analyze
        analysis_result = await Runner.run(
            starting_agent=create_analysis_agent(),
            input=question,
            context=content,
            session=session,
        )
        analysis_text = str(getattr(analysis_result, "final_output", "")).strip()

        # Communicate
        final_result = await Runner.run(
            starting_agent=create_communication_agent(),
            input=question,
            context=analysis_text,
            session=session,
        )

        print("\nAgent:")
        print(getattr(final_result, "final_output", ""))
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())