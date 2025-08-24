from opentelemetry import trace
import os
from time import perf_counter


class _TimedSpan:
    def __init__(self, tracer, name: str, attributes: dict | None = None):
        self._context_manager = tracer.start_as_current_span(name, attributes=attributes or {})
        self._current_span = None
        self._start_time = None

    def __enter__(self):
        token = self._context_manager.__enter__()
        self._current_span = trace.get_current_span()
        self._start_time = perf_counter()
        return token

    def __exit__(self, exception_type, exception_value, traceback):
        try:
            if self._current_span is not None and self._start_time is not None:
                duration_seconds = perf_counter() - self._start_time
                try:
                    self._current_span.set_attribute("duration.s", duration_seconds)
                    threshold = float(os.getenv("SLOW_TRACE_THRESHOLD_S", "5.0"))
                    if duration_seconds > threshold:
                        self._current_span.set_attribute("slow", True)
                        self._current_span.set_attribute("slow.threshold.s", threshold)
                except Exception:
                    pass
        finally:
            return self._context_manager.__exit__(exception_type, exception_value, traceback)


def start_reasoning_span(user_query: str, agent_name: str):
    tracer = trace.get_tracer(__name__)
    return _TimedSpan(
        tracer,
        "agent_reasoning",
        attributes={
            "agent.name": agent_name,
            "user.query": user_query,
        },
    )


def start_tool_span(tool_name: str, tool_input: str):
    tracer = trace.get_tracer(__name__)
    return _TimedSpan(
        tracer,
        "tool_execution",
        attributes={
            "tool.name": tool_name,
            "tool.input": tool_input,
        },
    )


def start_llm_span(
    model: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_cost: float | None = None,
):
    tracer = trace.get_tracer(__name__)
    attributes = {"llm.model": model}
    if prompt_tokens is not None:
        attributes["llm.prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        attributes["llm.completion_tokens"] = completion_tokens
    if total_cost is not None:
        attributes["llm.total_cost_usd"] = total_cost
    return _TimedSpan(tracer, "llm_call", attributes=attributes)


