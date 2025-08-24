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
    - PHOENIX_API_KEY (auth)
    - PHOENIX_COLLECTOR_ENDPOINT (fallback if endpoint arg missing)

    Returns (tracer, tracer_provider).
    """
    phoenix_api_key = os.getenv("PHOENIX_API_KEY")
    if not phoenix_api_key:
        raise ValueError("PHOENIX_API_KEY environment variable is required")
    
    print("🔗 Phoenix endpoint:", os.getenv("PHOENIX_COLLECTOR_ENDPOINT"))
    print("🔑 Phoenix key present:", bool(phoenix_api_key))
    
    tracer_provider = register(
        project_name=os.getenv("PHOENIX_PROJECT_NAME", "MyProject"),
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/v1/traces"),
        api_key=phoenix_api_key,
        batch=True,
    )
    
    tracer = tracer_provider.get_tracer(__name__)
    return tracer, tracer_provider
