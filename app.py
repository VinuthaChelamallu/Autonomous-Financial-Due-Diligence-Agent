from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from datetime import datetime, UTC  # ✅ new

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
from google.adk.memory import InMemoryMemoryService
from google.adk.models.google_llm import Gemini  # can be used later if needed
from google.genai import types  # still used for Content/Part

from agents.mas_root import build_mas
from agents.qa_agent_adk import qa_agent
# ❌ REMOVED: from agents.live_market_agent_adk import build_live_market_agent


# ---------------------------
# Config
# ---------------------------
APP_NAME = "agents"
USER_ID = "vinuthac"
SESSION_ID = "session_dynamic"

HTML_PATH = "C:/Users/chvin/OneDrive/Documents/google 3011/capstone_agent/data/Apple 2025.html"


# ---------------------------
# Helpers
# ---------------------------
def _print_per_period_metrics(state: Dict[str, Any]) -> None:
    """
    Debug helper for final MAS summary.

    Handles BOTH:
    - Old style: state["financial_metrics"] = {"per_period": {...}, "yoy": {...}}
    - New style: state["financial_metrics"] = { "2025-09-27": {..}, "2024-09-28": {..}, ... }
    - And prefers memo_financials["latest_metrics"] if present.
    """
    fm = state.get("financial_metrics") or {}
    memo_fin = state.get("memo_financials") or {}

    # Prefer key_period from memo_financials, fall back to top-level key_period
    key_period = memo_fin.get("key_period") or state.get("key_period")

    # ------------ 1) Figure out periods & per-period dict ------------
    periods: List[str] = []
    metrics_by_period: Dict[str, Dict[str, Any]] = {}

    if isinstance(fm, dict):
        # Old style: {"per_period": {...}, "yoy": {...}}
        if isinstance(fm.get("per_period"), dict):
            metrics_by_period = fm["per_period"]
            periods = list(metrics_by_period.keys())
        else:
            # New style: {"2025-09-27": {...}, "2024-09-28": {...}, ...}
            metrics_by_period = {
                p: v for p, v in fm.items() if isinstance(v, dict)
            }
            periods = list(metrics_by_period.keys())

    print("\n[Financial] periods:", periods)

    # ------------ 2) Prefer latest_metrics from memo_financials ------------
    latest_metrics = memo_fin.get("latest_metrics") or {}

    if latest_metrics and key_period:
        print(f"\n[Financial] Per-period metrics for {key_period}:")
        key_fields = [
            "revenue",
            "net_income",
            "operating_cash_flow",
            "capital_expenditures",
            "ebitda",
            "ebitda_margin",
            "net_margin",
            "fcf",
            "fcf_margin",
            "revenue_growth",
            "net_income_growth",
            "fcf_growth",
            "ebitda_growth",
        ]
        for k in key_fields:
            if k in latest_metrics:
                print(f"  {k}: {latest_metrics.get(k)}")
        return

    # ------------ 3) Fallback to per-period dict if latest_metrics missing ------------
    if key_period and key_period in metrics_by_period:
        m = metrics_by_period[key_period]
        print(f"\n[Financial] Per-period metrics for {key_period}:")
        key_fields = [
            "revenue",
            "net_income",
            "operating_cash_flow",
            "capital_expenditures",
            "ebitda",
            "ebitda_margin",
            "net_margin",
            "fcf",
            "fcf_margin",
        ]
        for k in key_fields:
            if k in m:
                print(f"  {k}: {m.get(k)}")
    else:
        print("\n[Financial] No per-period metrics found for key_period.")


def _print_risk_mdna_summary(state: Dict[str, Any]) -> None:
    risk_summary = state.get("risk_mdna_summary") or {}
    print("\n[Risk/MD&A] risk_mdna_summary keys:", list(risk_summary.keys()))

    # Prefer canonical "top_risks"
    top_risks = risk_summary.get("top_risks") or []
    headwinds = risk_summary.get("headwinds") or []
    tailwinds = risk_summary.get("tailwinds") or []
    print(
        f"[Risk/MD&A] #top_risks={len(top_risks)}, "
        f"#headwinds={len(headwinds)}, #tailwinds={len(tailwinds)}"
    )


def _print_memo_full(state: Dict[str, Any]) -> None:
    # ✅ Use new key written by MemoAgent (fall back to old one if present)
    memo_text = (
        state.get("investment_memo")
        or state.get("investment_memo_markdown")
        or ""
    )

    if not memo_text:
        print("\n[Memo] No memo found in state.")
        return

    company = state.get("company")
    key_period = state.get("key_period")

    print("\n" + "=" * 80)
    print("=== INVESTMENT MEMO (FULL) ===")
    print(f"[Memo] company={company!r}, key_period={key_period!r}")
    print("-" * 80)
    print(memo_text)
    print("=" * 80 + "\n")


def _print_qa_snippet(state: Dict[str, Any]) -> None:
    answer = state.get("qa_last_answer") or ""
    if answer:
        print("\n=== Q&A ANSWER (snippet) ===")
        print(answer[:600], "..." if len(answer) > 600 else "")


# ---------------------------
# Main
# ---------------------------
async def main() -> None:
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()

    # (Optional) Shared LLM instance if you later hook it via ctx.llm in agents
    llm = Gemini(model_name="gemini-2.0-flash-lite")

    # Create a brand-new session
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={"html_path": HTML_PATH},
    )

    # Build MAS pipeline (Extraction → Financial → Risk/MD&A → Memo → Company → Live → Valuation)
    mas = build_mas()

    runner = Runner(
        agent=mas,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
        # NOTE: this Runner version does not accept `llm=`
    )

    msg = types.Content(
        role="user",
        parts=[types.Part(text="Run full MAS pipeline.")],
    )

    print("=== Running MAS Pipeline ===")
    for ev in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=msg,
    ):
        print(f"[MAS Event] from {ev.author}")

    # Reload MAS state
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    state: Dict[str, Any] = session.state

    # -----------------------
    # Final state + memo print
    # -----------------------
    print("\n=== FINAL STATE SUMMARY (after MAS) ===")
    print("state keys:", list(state.keys()))
    print("company:", state.get("company"))
    print("available_periods:", state.get("available_periods"))
    print("key_period:", state.get("key_period"))

    _print_per_period_metrics(state)
    _print_risk_mdna_summary(state)
    _print_memo_full(state)

    # -----------------------
    # LIVE STOCK (canonical: Finnhub via LiveMarketStateAgent)
    # -----------------------
    # LiveMarketStateAgent (in MAS) should have already populated:
    #   state["live_quote"] and/or state["live_stock"]
    live_quote: Dict[str, Any] | None = state.get("live_quote")
    live_stock_raw: Dict[str, Any] | None = state.get("live_stock")

    if live_quote or live_stock_raw:
        quote = live_quote or live_stock_raw or {}
        print("\n=== LIVE STOCK (from Finnhub) ===")
        print(f"symbol: {quote.get('symbol')}")
        print(f"as_of: {quote.get('as_of')}")
        print(f"current_price: {quote.get('current_price')}")
        print(f"high_price: {quote.get('high_price')}")
        print(f"low_price: {quote.get('low_price')}")
        print(f"open_price: {quote.get('open_price')}")
        print(f"previous_close_price: {quote.get('previous_close_price')}")
        print(f"change: {quote.get('change')}")
        print(f"change_percent: {quote.get('change_percent')}")
    else:
        print("\n[Live] No live stock data found in state.")
        print("[Live] Check that LiveMarketStateAgent is wired into MAS and "
              "that FINNHUB_API_KEY is set.")

    # 🔁 Persist (no-op if already there, but keeps pattern consistent)
    await session_service.append_event(
        session,
        Event(
            author="system",
            actions=EventActions(
                state_delta={
                    "live_stock": state.get("live_stock"),
                    "live_quote": state.get("live_quote"),
                }
            ),
        ),
    )

    # -----------------------
    # Store memo in memory
    # -----------------------
    memo_text = (
        state.get("investment_memo")
        or state.get("investment_memo_markdown")
        or ""
    )
    if memo_text:
        await session_service.append_event(
            session,
            Event(
                author="memo_agent",
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=memo_text)],
                ),
            ),
        )
        print("\n🧠 Memo stored in memory")

    await memory_service.add_session_to_memory(session)
    print("\n✅ Session added to memory")

    # -----------------------
    # Q&A Chat Loop
    # -----------------------
    qa_runner = Runner(
        agent=qa_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
        # no llm kwarg
    )

    print("\n=== Q&A CHAT === (type quit to exit)\n")

    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in ("quit", "exit"):
            break

        await session_service.append_event(
            session,
            Event(
                author="user",
                actions=EventActions(state_delta={"qa_last_question": user_q}),
            ),
        )

        qa_msg = types.Content(role="user", parts=[types.Part(text=user_q)])

        for ev in qa_runner.run(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=qa_msg,
        ):
            pass

        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        _print_qa_snippet(session.state)


if __name__ == "__main__":
    asyncio.run(main())
