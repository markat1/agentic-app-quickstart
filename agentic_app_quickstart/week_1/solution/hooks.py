from typing import Any, List, Optional
import os
from time import perf_counter

import pandas as pd
from agents import RunHooks, Agent
from agents.run_context import RunContextWrapper


class FollowUpHooks(RunHooks):
    """
    Hook that captures follow-up suggestions after the analysis agent finishes.
    Suggestions are derived directly from the current DataFrame context to avoid recursion.
    """

    def __init__(self) -> None:
        self.last_suggestions: List[str] = []
        self._cursor: int = 0
        self._t0: Optional[float] = None
        self.last_duration_s: Optional[float] = None
        self.last_slow: bool = False

    async def on_agent_start(self, ctx: RunContextWrapper[Any], agent: Agent) -> None:
        if agent.name == "AnalysisAssistantAgent":
            self._t0 = perf_counter()

    async def on_agent_end(self, ctx: RunContextWrapper[Any], agent: Agent, output: Any) -> None:
        if agent.name != "AnalysisAssistantAgent":
            return
        # Duration tracking and slow-alert flag
        try:
            if self._t0 is not None:
                self.last_duration_s = perf_counter() - self._t0
            else:
                self.last_duration_s = None
            threshold = float(os.getenv("SLOW_TRACE_THRESHOLD_S", "5.0"))
            self.last_slow = bool(self.last_duration_s is not None and self.last_duration_s > threshold)
        except Exception:
            self.last_duration_s = None
            self.last_slow = False
        try:
            df = ctx.context if isinstance(ctx.context, pd.DataFrame) else None
            if df is None:
                self.last_suggestions = []
                self._cursor = 0
                return
               
            # Get column information for better suggestions
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            all_cols = list(df.columns)
           
            suggestions: List[str] = []
           
            # Basic data exploration suggestions
            suggestions.append("What columns are available in this dataset?")
           
            # Numeric column suggestions
            if len(numeric_cols) >= 2:
                x, y = numeric_cols[0], numeric_cols[1]
                suggestions.append(f"Create a scatter plot of {x} vs {y}")
                suggestions.append(f"What is the correlation between {x} and {y}?")
                suggestions.append(f"Find the average {x} and {y}")
            elif len(numeric_cols) == 1:
                c = numeric_cols[0]
                suggestions.append(f"Show a histogram of {c}")
                suggestions.append(f"What are the min, max, and average values for {c}?")
                suggestions.append(f"Are there any outliers in the {c} data?")
           
            # Categorical column suggestions
            if categorical_cols:
                for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
                    suggestions.append(f"How many unique values are in the {col} column?")
                    suggestions.append(f"What are the most common values in {col}?")
           
            # Cross-analysis suggestions
            if numeric_cols and categorical_cols:
                num_col = numeric_cols[0]
                cat_col = categorical_cols[0]
                suggestions.append(f"What is the average {num_col} for each {cat_col}?")
                suggestions.append(f"Compare {num_col} across different {cat_col} categories")
           
            # Advanced analysis suggestions
            if len(numeric_cols) >= 3:
                suggestions.append("Create a correlation matrix for all numeric columns")
                suggestions.append("Find outliers in the numeric columns")
           
            # Data quality suggestions
            suggestions.append("Are there any missing values in the dataset?")
            suggestions.append("What is the data type of each column?")
           
            # Limit suggestions and ensure uniqueness
            unique_suggestions = list(dict.fromkeys(suggestions))  # Remove duplicates while preserving order
            new_list = unique_suggestions[:8]  # Show up to 8 suggestions
           
            if new_list != self.last_suggestions:
                self.last_suggestions = new_list
                self._cursor = 0
        except Exception:
            self.last_suggestions = []
            self._cursor = 0

    def next_suggestion(self) -> Optional[str]:
        if not self.last_suggestions:
            return None
        suggestion = self.last_suggestions[self._cursor % len(self.last_suggestions)]
        self._cursor += 1
        return suggestion


def prompt_next_question(hooks: Optional[FollowUpHooks]) -> Optional[str]:
    """
    Prompt the user with the next suggested question if available; otherwise fall back.
    Returns the next question string, or None to exit.
    """
    if hooks is not None and hooks.last_slow and hooks.last_duration_s is not None:
        # Brief inline alert before proposing the next question
        print(f"Alert: last step took {hooks.last_duration_s:.1f}s; consider refining or scoping your question.")

    top = hooks.next_suggestion() if hooks is not None else None
    if top:
        user_input = input(f"User (Enter to ask: {top}): ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            return None
        return top if user_input == "" else user_input
    next_question = input("User: ").strip()
    if next_question.lower() in {"exit", "quit", "q"}:
        return None
    return next_question