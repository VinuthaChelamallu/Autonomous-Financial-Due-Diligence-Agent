# ui.py
from __future__ import annotations

import os
import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime as dt

import streamlit as st

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
from google.adk.memory import InMemoryMemoryService
from google.genai import types

from agents.mas_root import build_mas
from agents.qa_agent_adk import qa_agent


# ---------------------------
# Config (match app.py)
# ---------------------------
APP_NAME = "agents"
USER_ID = "vinuthac"
SESSION_ID = "session_dynamic"

DATA_DIR = "/workspaces/Finance-Agent/capstone_agent/data"
DEFAULT_HTML_PATH = "/workspaces/Finance-Agent/capstone_agent/data/walmart.html"


# ---------------------------
# Async helper
# ---------------------------
def run_coro(coro):
    """Run an async coroutine in a blocking way (Streamlit-safe pattern)."""
    return asyncio.run(coro)


# ---------------------------
# Number Formatting
# ---------------------------
def _format_money(value: Any) -> str:
    """Format large dollar amounts into human-readable form."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    sign = "-" if v < 0 else ""
    v = abs(v)

    if v >= 1_000_000_000:
        return f"{sign}{v / 1_000_000_000:,.2f} billion"
    elif v >= 1_000_000:
        return f"{sign}{v / 1_000_000:,.2f} million"
    else:
        return f"{sign}{v:,.0f}"


def _format_pct(value: Any) -> str:
    """Format numeric values as percentages, handling both 0.05 and 5 -> 5.0%."""
    # If it's already a string with %, just return it
    if isinstance(value, str) and "%" in value:
        return value

    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    # Heuristic: margins and growth often stored as 0.05 → 5%
    if -1.0 <= v <= 1.0:
        v = v * 100.0

    return f"{v:.1f}%"


# ---------------------------
# Init Services & Session
# ---------------------------
def init_adk_objects() -> None:
    if "session_service" in st.session_state:
        return

    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()
    mas_agent = build_mas()

    # Use uploaded file if present; otherwise default Walmart file
    html_path = st.session_state.get("html_path", DEFAULT_HTML_PATH)

    run_coro(
        session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
            state={"html_path": html_path},
        )
    )

    st.session_state.session_service = session_service
    st.session_state.memory_service = memory_service
    st.session_state.mas_agent = mas_agent


def get_state() -> Dict[str, Any]:
    session = run_coro(
        st.session_state.session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
    )
    return session.state


def persist_live_quote_and_memo(state: Dict[str, Any]) -> None:
    """Mirror what app.py does: persist live_quote/live_stock + memo into memory."""
    session_service: InMemorySessionService = st.session_state.session_service
    memory_service: InMemoryMemoryService = st.session_state.memory_service

    session = run_coro(
        session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
    )

    # Persist live stock info as a state_delta event (no-op if missing)
    run_coro(
        session_service.append_event(
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
    )

    # Store full memo as an Event from memo_agent
    memo_text = (
        state.get("investment_memo")
        or state.get("investment_memo_markdown")
        or ""
    )
    if memo_text:
        run_coro(
            session_service.append_event(
                session,
                Event(
                    author="memo_agent",
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=memo_text)],
                    ),
                ),
            )
        )

    # Add session to memory (for rolling summary / long-term context)
    run_coro(memory_service.add_session_to_memory(session))


# ---------------------------
# UI Helper Renderers (by agent)
# ---------------------------
def render_company_header(state: Dict[str, Any]) -> None:
    company = state.get("company", "Unknown company")
    ticker = state.get("ticker", state.get("symbol", ""))
    key_period = state.get("key_period")

    title_line = company
    if ticker:
        title_line += f" ({ticker})"

    st.header(title_line)
    if key_period:
        st.caption(f"Analysis period: **{key_period}**")


# ExtractionAgent tab
def render_extraction_section(state: Dict[str, Any]) -> None:
    company = state.get("company")
    ticker = state.get("ticker") or state.get("symbol")
    cik = state.get("cik")
    periods = state.get("available_periods") or []

    st.markdown("### Entity Details")

    cols = st.columns(3)
    cols[0].metric("Company", company if company else "-")
    cols[1].metric("Ticker", ticker if ticker else "-")
    cols[2].metric("CIK", cik if cik else "-")

    if periods:
        st.markdown("**Reporting periods detected:**")
        period_rows = [{"Period end date": p} for p in periods]

        import pandas as pd
        period_df = pd.DataFrame(period_rows)
        st.dataframe(period_df, hide_index=True)
    else:
        st.info("No reporting periods detected yet.")

    extraction_raw = state.get("extraction")
    if extraction_raw:
        with st.expander("Raw extraction summary (debug)"):
            st.json(extraction_raw)


# FinancialAgent tab
def render_financial_section(state: Dict[str, Any]) -> None:
    memo_fin = state.get("memo_financials") or {}
    latest_metrics: Dict[str, Any] = memo_fin.get("latest_metrics") or {}
    yoy_strings: Dict[str, Any] = (
        memo_fin.get("yoy_strings")
        or memo_fin.get("yoy_metrics")
        or {}
    )

    if not latest_metrics:
        st.info("No financial metrics found in state.")
        return

    # Map internal keys → user-friendly labels
    label_map = {
        "revenue": "Revenue (USD)",
        "net_income": "Net income (USD)",
        "operating_cash_flow": "Operating cash flow (USD)",
        "capital_expenditures": "Capital expenditures (USD)",
        "gross_margin_pct": "Gross margin (%)",
        "operating_margin_pct": "Operating margin (%)",
        "ebitda_margin_pct": "EBITDA margin (%)",
        "net_margin_pct": "Net margin (%)",
        "fcf": "Free cash flow (USD)",
        "fcf_margin_pct": "Free cash flow margin (%)",
        "revenue_growth_pct": "Revenue growth year-over-year",
        "net_income_growth_pct": "Net income growth year-over-year",
        "fcf_growth_pct": "Free cash flow growth year-over-year",
        "ebitda_growth_pct": "EBITDA growth year-over-year",
    }

    # Keys we want to highlight as "almost necessary"
    core_keys = [
        "revenue",
        "revenue_growth_pct",
        "net_income",
        "net_income_growth_pct",
        "operating_cash_flow",
        "fcf",
        "fcf_margin_pct",
        "net_margin_pct",
        "ebitda_margin_pct",
    ]

    # Which keys are money vs percentages
    amount_keys = {
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capital_expenditures",
        "fcf",
    }

    percent_keys = {
        "gross_margin_pct",
        "operating_margin_pct",
        "ebitda_margin_pct",
        "net_margin_pct",
        "fcf_margin_pct",
        "revenue_growth_pct",
        "net_income_growth_pct",
        "fcf_growth_pct",
        "ebitda_growth_pct",
    }

    core_rows: List[Dict[str, Any]] = []
    extra_rows: List[Dict[str, Any]] = []

    # Latest point-in-time metrics
    for key, value in latest_metrics.items():
        label = label_map.get(key, key.replace("_", " ").title())

        # Format based on data type
        if key in amount_keys:
            display_value = _format_money(value)
        elif key in percent_keys:
            display_value = _format_pct(value)
        else:
            display_value = value

        # Force to string to keep Arrow happy
        row = {"Metric": label, "Value": str(display_value)}

        if key in core_keys:
            core_rows.append(row)
        else:
            extra_rows.append(row)

    # YoY metrics – some are core, some are additional
    for key, value in yoy_strings.items():
        label = label_map.get(key, key.replace("_", " ").title())

        if key in percent_keys:
            display_value = _format_pct(value)
        else:
            display_value = value

        # Force to string again
        row = {"Metric": label, "Value": str(display_value)}

        if key in ["revenue_growth_pct", "net_income_growth_pct"]:
            core_rows.append(row)
        else:
            extra_rows.append(row)

    import pandas as pd

    if core_rows:
        st.markdown("**Key metrics**")
        core_df = pd.DataFrame(core_rows)
        st.dataframe(core_df, hide_index=True)
    else:
        st.info("No key financial rows available to display.")

    if extra_rows:
        with st.expander("Additional metrics"):
            extra_df = pd.DataFrame(extra_rows)
            st.dataframe(extra_df, hide_index=True)


def _numbered_table(items: List[Dict[str, Any]], field_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Convert a list of dicts into a numbered table.
    - Numbering starts from 1 (not 0)
    - field_map: {source_key: human_label}
    """
    table_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        row: Dict[str, Any] = {"No.": idx}
        for src_key, label in field_map.items():
            row[label] = item.get(src_key)
        table_rows.append(row)
    return table_rows


# RiskMdnaAgent tab
def render_risks_section(state: Dict[str, Any]) -> None:
    risk_summary = state.get("risk_mdna_summary") or {}
    top_risks = risk_summary.get("top_risks") or []
    headwinds = risk_summary.get("headwinds") or []
    tailwinds = risk_summary.get("tailwinds") or []

    # Top risks
    st.markdown("##### Key Risks (Item 1A)")
    if top_risks:
        risk_table = _numbered_table(
            top_risks,
            {
                "name": "Risk name",
                "category": "Category",
                "description": "Description",
                "financial_impact": "Financial impact",
                "likelihood": "Likelihood",
            },
        )
        import pandas as pd
        df_risks = pd.DataFrame(risk_table)
        st.dataframe(df_risks, hide_index=True)
    else:
        st.info("No structured key risks were extracted.")

    # Headwinds
    st.markdown("##### Headwinds (Challenges)")
    if headwinds:
        headwind_table = _numbered_table(
            headwinds,
            {
                "name": "Headwind",
                "description": "Description",
            },
        )
        import pandas as pd
        df_headwinds = pd.DataFrame(headwind_table)
        st.dataframe(df_headwinds, hide_index=True)
    else:
        st.info("No headwinds extracted.")

    # Tailwinds
    st.markdown("##### Tailwinds (Supportive Factors)")
    if tailwinds:
        tailwind_table = _numbered_table(
            tailwinds,
            {
                "name": "Tailwind",
                "description": "Description",
            },
        )
        import pandas as pd
        df_tailwinds = pd.DataFrame(tailwind_table)
        st.dataframe(df_tailwinds, hide_index=True)
    else:
        st.info("No tailwinds extracted.")


# CompanyAgent tab
def render_company_overview(state: Dict[str, Any]) -> None:
    company = state.get("company", "")

    if company:
        st.markdown(f"## {company}")

    overview = state.get("company_overview")
    if overview:
        st.markdown(overview)
    else:
        st.info("No company overview found in state.")


# MarketAnalysisAgent tab
def render_market_analysis(state: Dict[str, Any]) -> None:
    raw_market = state.get("market_analysis")

    # Nothing in state
    if not raw_market:
        st.info("No market analysis found in state.")
        return

    market: Dict[str, Any] = {}

    # If it's already a dict (ideal case)
    if isinstance(raw_market, dict):
        market = raw_market
    # If it's a string, try to parse JSON; otherwise show as plain text
    elif isinstance(raw_market, str):
        parsed = None
        try:
            parsed = json.loads(raw_market)
        except Exception:
            # maybe it's a longer LLM answer with a JSON block inside; try to grab first {...}
            try:
                start = raw_market.find("{")
                end = raw_market.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = raw_market[start : end + 1]
                    parsed = json.loads(candidate)
            except Exception:
                parsed = None

        if isinstance(parsed, dict):
            market = parsed
        else:
            # graceful fallback: show whatever we have and exit
            st.markdown("### Market analysis (raw text)")
            st.write(raw_market)
            return
    else:
        st.info("Market analysis is in an unexpected format.")
        st.json(raw_market)
        return

    # -------------------
    # Industry
    # -------------------
    industry = market.get("industry")
    if industry:
        st.markdown(f"**Industry:** {industry}")

    # -------------------
    # Growth outlook
    # -------------------
    st.markdown("### Growth outlook")
    growth = market.get("growth")

    if isinstance(growth, dict):
        cagr = growth.get("cagr_percent") or growth.get("CAGR")
        period = growth.get("period")

        summary_bits = []
        if cagr and cagr != "unknown":
            summary_bits.append(f"Estimated CAGR: **{cagr}**")
        if period and period != "unknown":
            summary_bits.append(f"over **{period}**")

        if summary_bits:
            st.markdown(" ".join(summary_bits))

        drivers = growth.get("drivers") or []
        if drivers:
            st.markdown("**Key growth drivers:**")
            st.markdown("\n".join(f"- {d}" for d in drivers))

        headwinds = growth.get("headwinds") or []
        if headwinds:
            st.markdown("**Growth headwinds:**")
            st.markdown("\n".join(f"- {h}" for h in headwinds))
    elif growth:
        # If the agent returned a simple string
        st.write(growth)
    else:
        st.info("No structured growth outlook available.")

    # -------------------
    # Competitive landscape
    # -------------------
    comp = market.get("competitive_landscape") or {}
    if comp:
        st.markdown("### Competitive landscape")

        key_competitors = comp.get("key_competitors") or []
        if key_competitors:
            st.markdown("**Key competitors:**")
            for c in key_competitors:
                name = c.get("name") or "Competitor"
                share = c.get("approx_market_share_percent")
                positioning = c.get("positioning")

                header = f"- **{name}**"
                if share and share != "unknown":
                    header += f" — approx. market share: {share}"
                st.markdown(header)
                if positioning:
                    st.markdown(f"  - {positioning}")

        moats = comp.get("moats_and_differentiation") or []
        if moats:
            st.markdown("**Moats and differentiation:**")
            st.markdown("\n".join(f"- {m}" for m in moats))

        threats = comp.get("threats_from_competition") or []
        if threats:
            st.markdown("**Threats from competition:**")
            st.markdown("\n".join(f"- {t}" for t in threats))

    # -------------------
    # Industry trends
    # -------------------
    trends = market.get("industry_trends") or {}
    if trends:
        st.markdown("### Industry trends")

        tailwinds = trends.get("tailwinds") or []
        if tailwinds:
            st.markdown("**Tailwinds:**")
            st.markdown("\n".join(f"- {tw}" for tw in tailwinds))

        structural = trends.get("structural_shifts") or []
        if structural:
            st.markdown("**Structural shifts:**")
            st.markdown("\n".join(f"- {s}" for s in structural))

        risks = trends.get("risks") or []
        if risks:
            st.markdown("**Risks:**")
            st.markdown("\n".join(f"- {r}" for r in risks))

    # -------------------
    # Sources / references (optional)
    # -------------------
    sources = market.get("sources")
    if sources:
        with st.expander("Sources / references"):
            st.json(sources)


# LiveMarketStateAgent tab
def render_live_stock(state: Dict[str, Any]) -> None:
    live_quote: Dict[str, Any] | None = state.get("live_quote")
    live_stock_raw: Dict[str, Any] | None = state.get("live_stock")

    quote = live_quote or live_stock_raw or {}

    if not quote:
        st.info(
            "No live stock data found. "
            "Check that LiveMarketStateAgent is wired into MAS and FINNHUB_API_KEY is set."
        )
        return

    cols = st.columns(4)
    cols[0].metric("Symbol", quote.get("symbol"))
    cols[1].metric("Current price", quote.get("current_price"))
    cols[2].metric("Daily change", quote.get("change"))
    cols[3].metric("Change (%)", quote.get("change_percent"))
    st.caption(f"As of: {quote.get('as_of')}")
    st.write(
        f"High: {quote.get('high_price')}, "
        f"Low: {quote.get('low_price')}, "
        f"Open: {quote.get('open_price')}, "
        f"Previous close: {quote.get('previous_close_price')}"
    )


# ValuationAgent tab
def render_valuation_section(state: Dict[str, Any]) -> None:
    valuation = state.get("valuation_metrics") or {}

    if not valuation:
        st.info("No valuation metrics found in state.")
        return

    cols = st.columns(3)

    market_cap = valuation.get("market_cap")
    pe = valuation.get("pe")
    fcf_yield_val = valuation.get("fcf_yield_pct") or valuation.get("fcf_yield")

    # Use the same helpers as Financials
    market_cap_display = _format_money(market_cap)

    if isinstance(pe, (int, float)):
        pe_display = f"{pe:.2f}"
    else:
        pe_display = str(pe) if pe is not None else "-"

    if fcf_yield_val is not None:
        fcf_yield_display = _format_pct(fcf_yield_val)
    else:
        fcf_yield_display = "-"

    cols[0].metric("Market capitalization", market_cap_display)
    cols[1].metric("Price / Earnings ratio", pe_display)
    cols[2].metric("Free cash flow yield", fcf_yield_display)

    with st.expander("Full valuation payload"):
        st.json(valuation)


# MemoAgent tab
def render_investment_memo(state: Dict[str, Any]) -> None:
    memo_text = (
        state.get("investment_memo")
        or state.get("investment_memo_markdown")
        or ""
    )

    if memo_text:
        st.markdown(memo_text)
    else:
        st.info("No investment memo found in state.")


# Memory & Session tab (state + rolling summary)
def render_memory_session_section(state: Optional[Dict[str, Any]] = None) -> None:
    st.subheader("Memory / State View")

    if not state:
        st.info("Run the pipeline or ask a question first to populate state & memory.")
        return

    memo = (
        state.get("investment_memo")
        or state.get("investment_memo_markdown")
    )
    last_q = state.get("qa_last_question")
    last_a = state.get("qa_last_answer")
    conv_summary = state.get("conversation_summary")
    qa_history = state.get("qa_history") or []

    with st.expander("Investment Memo stored in state"):
        if memo:
            st.markdown(memo)
        else:
            st.write("No memo found in state.")

    with st.expander("Last Q&A stored in state"):
        st.write(f"**Last Question:** {last_q or '—'}")
        st.write(f"**Last Answer:** {last_a or '—'}")

    with st.expander("Rolling Conversation Summary"):
        if conv_summary:
            st.markdown(conv_summary)
        else:
            st.write("No conversation summary yet.")

    with st.expander("Compact Q&A History (last 2 turns)"):
        if isinstance(qa_history, list) and qa_history:
            for i, entry in enumerate(qa_history, start=1):
                q = entry.get("q") or ""
                a = entry.get("a") or ""
                route = entry.get("route") or "unknown"
                st.markdown(f"**Turn {i}**  \n_Q ({route}):_ {q}  \n_A:_ {a}")
        else:
            st.write("No Q&A history yet.")


# ---------------------------
# Q&A Section (QAAgent)
# ---------------------------
def render_qa_section() -> None:
    st.subheader("Q&A Agent")

    # Use a form so that pressing Enter submits the question
    with st.form("qa_form"):
        user_q = st.text_input(
            "Ask a question about this company",
            key="qa_question",
        )
        submitted = st.form_submit_button("Ask")

    # If form not submitted, do nothing yet
    if not submitted:
        return

    # If submitted but empty question
    if not user_q or not user_q.strip():
        st.warning("Please enter a question.")
        return

    session_service: InMemorySessionService = st.session_state.session_service
    memory_service: InMemoryMemoryService = st.session_state.memory_service

    # Load session
    session = run_coro(
        session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
    )

    # Store last question in state (same pattern as app.py)
    run_coro(
        session_service.append_event(
            session,
            Event(
                author="user",
                actions=EventActions(
                    state_delta={"qa_last_question": user_q}
                ),
            ),
        )
    )

    qa_msg = types.Content(
        role="user",
        parts=[types.Part(text=user_q)],
    )

    qa_runner = Runner(
        agent=qa_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
    )

    # Run QA agent
    for _ in qa_runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=qa_msg,
    ):
        pass

    # Reload state and show the answer snippet
    state = get_state()
    st.session_state["state"] = state  # keep snapshot in sync for Memory tab
    answer = state.get("qa_last_answer") or "No answer generated."
    st.markdown("**Answer:**")
    st.write(answer)


# ---------------------------
# Main Streamlit app
# ---------------------------
def main():
    st.set_page_config(
        page_title="Autonomous Financial Due Diligence Agent",
        layout="wide",
    )

    st.title("Autonomous Financial Due Diligence Agent")
    st.caption(
        "Multi-Agent System for Automated 10-K Analysis, Live Market Intelligence, and Q&A"
    )

    # ----- Sidebar: file upload -----
    st.sidebar.header("Upload 10-K Document")

    uploaded = st.sidebar.file_uploader(
        "",
        type=["html", "htm", "xhtml"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        uploads_dir = os.path.join(DATA_DIR, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        safe_name = uploaded.name.replace(" ", "_")
        ts = dt.now().strftime("%Y%m%d_%H%M%S")
        dest_path = os.path.join(uploads_dir, f"{ts}_{safe_name}")

        with open(dest_path, "wb") as f:
            f.write(uploaded.read())

        st.session_state["html_path"] = dest_path
        st.sidebar.success("File uploaded, run the MAS pipeline.")

    # Init ADK objects after potential html_path change
    init_adk_objects()

    if st.sidebar.button("Run MAS pipeline for uploaded file"):
        # Before running MAS, update the session's html_path with the current selection
        html_path = st.session_state.get("html_path", DEFAULT_HTML_PATH)

        session_service: InMemorySessionService = st.session_state.session_service
        session = run_coro(
            session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=SESSION_ID,
            )
        )
        run_coro(
            session_service.append_event(
                session,
                Event(
                    author="system",
                    actions=EventActions(
                        state_delta={"html_path": html_path}
                    ),
                ),
            )
        )

        # Run the full MAS pipeline once
        mas_runner = Runner(
            agent=st.session_state.mas_agent,
            app_name=APP_NAME,
            session_service=st.session_state.session_service,
            memory_service=st.session_state.memory_service,
        )

        msg = types.Content(
            role="user",
            parts=[types.Part(text="Run full MAS pipeline.")],
        )

        # Consume all events
        for _ in mas_runner.run(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=msg,
        ):
            pass

        # Save latest state in session_state
        state = get_state()
        st.session_state["state"] = state

        # Persist live quote + memo into memory (like app.py)
        persist_live_quote_and_memo(state)

        st.sidebar.success("MAS pipeline completed and state updated.")

    # If we have state from a previous MAS run, render the dashboard
    state: Dict[str, Any] = st.session_state.get("state") or {}

    if not state:
        st.warning("Run the MAS pipeline to populate the data.")

    # Global header (company, ticker, period)
    render_company_header(state)

    # ---- Tabs per agent in requested order ----
    (
        tab_company,
        tab_extraction,
        tab_financials,
        tab_risks,
        tab_market,
        tab_live,
        tab_valuation,
        tab_memo,
        tab_qa,
        tab_memory,
    ) = st.tabs(
        [
            "Company Overview",
            "Extraction",
            "Financial Metrics",
            "Risks & MD&A",
            "Market Analysis",
            "Live Stock Price",
            "Valuation",
            "Investment Memo",
            "Q&A",
            "Memory and Session",
        ]
    )

    with tab_company:
        render_company_overview(state)

    with tab_extraction:
        render_extraction_section(state)

    with tab_financials:
        render_financial_section(state)

    with tab_risks:
        render_risks_section(state)

    with tab_market:
        render_market_analysis(state)

    with tab_live:
        render_live_stock(state)

    with tab_valuation:
        render_valuation_section(state)

    with tab_memo:
        render_investment_memo(state)

    with tab_qa:
        render_qa_section()

    with tab_memory:
        # Always fetch fresh session state for Memory tab
        live_state = get_state()
        render_memory_session_section(live_state)


if __name__ == "__main__":
    main()
