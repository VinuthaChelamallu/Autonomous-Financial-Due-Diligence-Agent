# server.py
"""
FastAPI wrapper to expose the ADK Multi-Agent System as a web service.

This makes the project *deployment ready* for Cloud Run / Agent Engine style
cloud runtimes, without changing your existing MAS logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.memory import InMemoryMemoryService
from google.adk.models.google_llm import Gemini

from agents.mas_root import build_mas


APP_NAME = "agents"          # same as your app.py
DEFAULT_USER_ID = "vinuthac"  # can be anything for now
DEFAULT_SESSION_ID = "session_api"


# ---------------------------
# Request / Response Models
# ---------------------------

class AnalyzeRequest(BaseModel):
    """
    Request body for running the MAS pipeline.

    Example:
    {
      "html_path": "data/Apple 2021.html",
      "question": "What are the main risks?"
    }
    """
    html_path: str
    question: Optional[str] = None  # for QA agent / chat


class AnalyzeResponse(BaseModel):
    success: bool
    message: str
    state: Dict[str, Any]


# ---------------------------
# Runner + MAS helpers
# ---------------------------

def create_runner() -> Runner:
    """
    Build an ADK Runner with in-memory session and memory services.

    This mirrors the pattern you're already using in app.py / app_session.py,
    but in a form that is easy to run in a cloud container.
    """
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()

    # Uses GOOGLE_API_KEY from the environment (in a real cloud runtime).
    model_provider = Gemini()

    runner = Runner(
        app_name=APP_NAME,
        model_provider=model_provider,
        session_service=session_service,
        memory_service=memory_service,
    )
    return runner


def run_mas_pipeline(
    html_path: str,
    question: Optional[str] = None,
    user_id: str = DEFAULT_USER_ID,
    session_id: str = DEFAULT_SESSION_ID,
) -> Dict[str, Any]:
    """
    Invoke your full MAS: Extraction -> Financials -> Risk/MD&A -> Memo -> Q&A.

    Initial state:
      - html_path (required)
      - qa_last_question (optional, for QA agent routing)

    All the other keys (company, ticker, financial_metrics, risk_mdna_summary,
    memo, qa_history, etc.) will be populated by the agents as usual.
    """
    runner = create_runner()
    mas = build_mas()

    init_state: Dict[str, Any] = {
        "html_path": html_path,
        "user_id": user_id,
        "session_id": session_id,
    }

    if question:
        init_state["qa_last_question"] = question

    # Synchronous invocation of the MAS graph
    final_state = runner.invoke(mas, init_state)
    return final_state


# ---------------------------
# FastAPI app
# ---------------------------

app = FastAPI(
    title="Financial Due Diligence Agent",
    description=(
        "Google ADK-based multi-agent system wrapped as an HTTP API. "
        "This service shape is deployment-ready for Cloud Run / Agent Engine."
    ),
    version="1.0.0",
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "message": "MAS service is up"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run the full pipeline on a 10-K HTML/iXBRL file and optional question.

    Example request:
      POST /analyze
      {
        "html_path": "data/Apple 2021.html",
        "question": "What are the key risks?"
      }
    """
    try:
        state = run_mas_pipeline(
            html_path=req.html_path,
            question=req.question,
        )

        return AnalyzeResponse(
            success=True,
            message="Pipeline executed successfully",
            state=state,
        )

    except Exception as e:
        # For production, you'd log this; for the capstone, this shape is enough.
        return AnalyzeResponse(
            success=False,
            message=f"Error running pipeline: {e}",
            state={},
        )
