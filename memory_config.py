from __future__ import annotations

import os
from typing import Optional

from google.adk.memory import VertexAiMemoryBankService


def get_vertex_memory_service() -> Optional[VertexAiMemoryBankService]:
    """
    Try to create a VertexAiMemoryBankService for long-term memory.

    It reads configuration from environment variables:

      - GOOGLE_CLOUD_PROJECT            (required)
      - GOOGLE_CLOUD_LOCATION           (optional, defaults to "us-central1")
      - GOOGLE_CLOUD_AGENT_ENGINE_ID    (required)

    If any required value is missing or something goes wrong,
    this returns None and prints a clear debug message, so your
    agent can still run *without* memory.
    """
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    agent_engine_id = os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID")

    if not project or not agent_engine_id:
        print(
            "[Memory] Vertex AI Memory Bank is NOT configured. "
            "Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_AGENT_ENGINE_ID "
            "to enable it."
        )
        return None

    try:
        memory_service = VertexAiMemoryBankService(
            project=project,
            location=location,
            agent_engine_id=agent_engine_id,
        )
        print(
            f"[Memory] VertexAiMemoryBankService created "
            f"(project={project}, location={location}, "
            f"agent_engine_id={agent_engine_id})."
        )
        return memory_service
    except Exception as exc:
        print(
            "[Memory] ERROR creating VertexAiMemoryBankService: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
