from __future__ import annotations

import os
from typing import Tuple
from phoenix.otel import register

def init_tracing(
    project_name: str | None = None,
    auto_instrument: bool = True,
    endpoint: str | None = None,
) -> Tuple[object, object]:
    """
    Initialize Phoenix OpenTelemetry tracing once at startup.

    Follows Phoenix docs: https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/setup-using-phoenix-otel

    Env used:
    - PHOENIX_COLLECTOR_ENDPOINT (fallback if endpoint arg missing)

    Returns (tracer, tracer_provider).
    """
    # Use provided endpoint or fall back to environment variable or default
    phoenix_endpoint = endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    
    print("🔗 Phoenix endpoint:", phoenix_endpoint)
    
    tracer_provider = register(
        project_name=project_name or os.getenv("PHOENIX_PROJECT_NAME", "default-project-name"),
        endpoint=phoenix_endpoint,
        protocol="http/protobuf",
        auto_instrument=auto_instrument,
        batch=True,
    )
    
    tracer = tracer_provider.get_tracer(__name__)
    return tracer, tracer_provider
