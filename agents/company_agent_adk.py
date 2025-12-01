# agents/company_agent_adk.py
from __future__ import annotations

from typing import AsyncGenerator, Dict, Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from google import genai

DEBUG_COMPANY_AGENT = True

# Gemini client (picks up GOOGLE_API_KEY from env)
_GEMINI_CLIENT = genai.Client()
_GEMINI_MODEL_COMPANY = "gemini-2.0-flash-lite"


class CompanyAgent(BaseAgent):
    """
    LLM-powered CompanyAgent.

    Responsibilities:
    - Read qualitative text from the 10-K (Item 1, MD&A) from state.
    - Summarize it into a memo-ready "Company Overview" section.
    - Store the markdown overview in state["company_overview"].
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        # ADK pattern: work with ctx.session.state
        state: Dict[str, Any] = ctx.session.state

        if DEBUG_COMPANY_AGENT:
            print("\n[CompanyAgent] START")
            print("[CompanyAgent] state keys BEFORE:", list(state.keys()))

        # --- 1) Pull inputs from state ---

        company = state.get("company") or state.get("company_name") or "The company"
        ticker = state.get("ticker") or state.get("symbol") or ""

        # Item 1 text is ideal; MD&A is used as backup context
        item_1_text: str = state.get("item_1_text") or ""
        mdna_text: str = state.get("mdna_text") or ""

        # --- 2) Handle missing context gracefully ---

        if not item_1_text and not mdna_text:
            if DEBUG_COMPANY_AGENT:
                print("[CompanyAgent] No Item 1 or MD&A text found. Using fallback overview.")

            overview = (
                f"## Company Overview\n\n"
                f"{company} ({ticker}) is a public company. "
                "The 10-K text for Item 1 / MD&A was not available in the parsed state, "
                "so a detailed overview could not be generated."
            )

            state_delta = {"company_overview": overview}

            if DEBUG_COMPANY_AGENT:
                print("[CompanyAgent] END (fallback), company_overview set.\n")

            # Yield event so MAS persists the new field
            yield Event(
                author=self.name,
                actions=EventActions(state_delta=state_delta),
            )
            return

        # --- 3) Build context text for the LLM ---

        context_chunks = []
        if item_1_text:
            # Truncate so we don't blow up token limits
            context_chunks.append("=== ITEM 1: BUSINESS DESCRIPTION ===\n" + item_1_text[:12000])
        if mdna_text:
            context_chunks.append("=== ITEM 7: MD&A (EXCERPT) ===\n" + mdna_text[:8000])

        context_text = "\n\n".join(context_chunks)

        # --- 4) Build prompt for Company Overview ---

        prompt = f"""
You are an equity research analyst.

Using ONLY the context from the company's 10-K filing below, write a concise,
bullet-style **Company Overview** section for an investment memo.

Company name: {company}
Ticker: {ticker or "N/A"}

Required structure (markdown):

## Company Overview

- **Business model:** ...
- **Products & services:** ...
- **Revenue segments / geographies (if mentioned):** ...
- **Target customers / end markets:** ...
- **Competitive advantages / moat:** ...
- **Recent strategic themes (if any):** ...

Rules:
- Base everything ONLY on the provided 10-K text.
- If something is not in the text, say "Not explicitly disclosed in the filing."
- Do not invent market size numbers or specific competitors that are not mentioned.
- Keep total length about 6–10 bullets.

10-K Context:
{context_text}
""".strip()

        if DEBUG_COMPANY_AGENT:
            print("[CompanyAgent] Calling Gemini for company_overview...")

        # --- 5) Call Gemini ---

        resp = _GEMINI_CLIENT.models.generate_content(
            model=_GEMINI_MODEL_COMPANY,
            contents=[prompt],
        )

        overview_text = getattr(resp, "text", "") or ""
        overview_text = overview_text.strip()

        # Fallback if something went wrong with LLM output
        if not overview_text:
            if DEBUG_COMPANY_AGENT:
                print("[CompanyAgent] Empty LLM response, using fallback overview.")

            overview_text = (
                f"## Company Overview\n\n"
                f"{company} ({ticker}) operates in its respective industry. "
                "A detailed overview could not be generated due to missing or invalid LLM output."
            )

        if DEBUG_COMPANY_AGENT:
            print("[CompanyAgent] company_overview (first 400 chars):")
            print(overview_text[:400])
            print("[CompanyAgent] END\n")

        state_delta = {"company_overview": overview_text}

        # --- 6) Yield event so MAS persists state_delta ---

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )


# 👇 THIS is what mas_root.py imports
company_agent = CompanyAgent(
    name="company_agent",
    description=(
        "Summarizes the business model, products, segments and moat into a "
        "memo-ready Company Overview section based on 10-K text."
    ),
)
