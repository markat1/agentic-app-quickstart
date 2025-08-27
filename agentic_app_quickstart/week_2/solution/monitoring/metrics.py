"""Lightweight helpers for timing and annotating spans with OpenTelemetry."""

from opentelemetry import trace
import os
from time import perf_counter, time as wall_time
from datetime import datetime, timezone
from typing import Any, Dict


class TimedSpan:
    """Context manager that records span duration and tags slow spans.

    - Sets "duration.s"
    - If duration exceeds SLOW_TRACE_THRESHOLD_S (default 5.0), sets "slow" and "slow.threshold.s"
    """

    def __init__(self, tracer, name: str, attributes: Dict[str, Any] | None = None):
        self._context_manager = tracer.start_as_current_span(name, attributes=attributes or {})
        self._current_span = None
        self._start_time = None
        self._start_walltime = None

    def __enter__(self):
        token = self._context_manager.__enter__()
        self._current_span = trace.get_current_span()
        self._start_time = perf_counter()
        self._start_walltime = wall_time()
        # Record wall-clock start time (UTC ISO8601) for easier correlation
        try:
            self._current_span.set_attribute("start.time", datetime.fromtimestamp(self._start_walltime, tz=timezone.utc).isoformat())
        except Exception:
            pass
        return token

    def __exit__(self, exception_type, exception_value, traceback):
        if self._current_span is not None and self._start_time is not None:
            duration_seconds = perf_counter() - self._start_time
            duration_milliseconds = duration_seconds * 1000.0
            self._current_span.set_attribute("duration.s", duration_seconds)
            self._current_span.set_attribute("duration.ms", duration_milliseconds)
            # Record wall-clock end time
            try:
                end_walltime = wall_time()
                self._current_span.set_attribute("end.time", datetime.fromtimestamp(end_walltime, tz=timezone.utc).isoformat())
            except Exception:
                pass
            threshold = float(os.getenv("SLOW_TRACE_THRESHOLD_S", "5.0"))
            if duration_seconds > threshold:
                self._current_span.set_attribute("slow", True)
                self._current_span.set_attribute("slow.threshold.s", threshold)
        return self._context_manager.__exit__(exception_type, exception_value, traceback)


def start_span(name: str, attributes: Dict[str, Any] | None = None):
    """Start a timed span with optional attributes."""
    tracer = trace.get_tracer(__name__)
    return TimedSpan(tracer, name, attributes=attributes)


def start_reasoning_span(user_query: str, agent_name: str):
    """Span for agent internal reasoning."""
    return start_span(
        "agent_reasoning",
        attributes={
            "agent.name": agent_name,
            "user.query": user_query,
        },
    )


def start_tool_span(tool_name: str, tool_input: str):
    """Span for tool/function execution with basic inputs."""
    return start_span(
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
    """Span for LLM calls with token and cost annotations (when available)."""
    attributes: Dict[str, Any] = {"llm.model": model}
    if prompt_tokens is not None:
        attributes["llm.prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        attributes["llm.completion_tokens"] = completion_tokens
    if total_cost is not None:
        attributes["llm.total_cost_usd"] = total_cost
    return start_span("llm_call", attributes=attributes)


