# agents/extraction_agent_adk.py
from __future__ import annotations

from typing import AsyncGenerator, Dict, Any, Set, List

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from tools.adk_pdf_tools import parse_10k_html_ixbrl

from bs4 import BeautifulSoup  # 🔧 new: fallback direct iXBRL parsing

DEBUG_EXTRACTION_AGENT = True


def _extract_revenue_direct_from_file(doc_path: str) -> Dict[str, float]:
    """
    Fallback: directly parse the 10-K HTML/iXBRL file to reconstruct
    the revenue series from ix:nonFraction facts.

    We look for:
      - context elements to map contextRef -> period end date
      - ix:nonFraction with name in REVENUE_CONCEPTS

    Returns:
      { "YYYY-MM-DD": scaled_revenue_value }
    """
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            html = f.read()
    except Exception as e:
        if DEBUG_EXTRACTION_AGENT:
            print(f"[ExtractionAgent] _extract_revenue_direct_from_file: failed to read {doc_path!r}: {e}")
        return {}

    soup = BeautifulSoup(html, "lxml")

    # 1) Build contextRef -> endDate map
    context_end_by_id: Dict[str, str] = {}

    def _is_context_tag(tag) -> bool:
        name = (tag.name or "").lower()
        # Covers xbrli:context, context, etc.
        return name.endswith("context")

    def _is_period_tag(tag) -> bool:
        name = (tag.name or "").lower()
        return name.endswith("period")

    def _is_end_or_instant_tag(tag) -> bool:
        name = (tag.name or "").lower()
        return name.endswith("enddate") or name.endswith("instant")

    for ctx in soup.find_all(_is_context_tag):
        ctx_id = ctx.get("id")
        if not ctx_id:
            continue

        period_tag = ctx.find(_is_period_tag)
        if not period_tag:
            continue

        end_or_instant = period_tag.find(_is_end_or_instant_tag)
        if not end_or_instant or not end_or_instant.text:
            continue

        period_str = end_or_instant.text.strip()
        if period_str:
            context_end_by_id[ctx_id] = period_str

    if DEBUG_EXTRACTION_AGENT:
        print(f"[ExtractionAgent] _extract_revenue_direct_from_file: #contexts={len(context_end_by_id)}")

    # 2) Scan ix:nonFraction revenue facts
    def _is_nonfraction_tag(tag) -> bool:
        name = (tag.name or "").lower()
        # Covers ix:nonfraction, nonfraction, etc.
        return name.endswith("nonfraction")

    REVENUE_CONCEPTS = {
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    }

    revenue_by_period: Dict[str, float] = {}

    for fact in soup.find_all(_is_nonfraction_tag):
        name = str(fact.get("name") or "").strip()
        if name not in REVENUE_CONCEPTS:
            continue

        ctx_ref = fact.get("contextref") or fact.get("contextRef")
        if not ctx_ref:
            continue

        period_str = context_end_by_id.get(ctx_ref)
        if not period_str:
            continue

        raw_text = (fact.text or "").strip().replace(",", "")
        if not raw_text:
            continue

        try:
            raw_val = float(raw_text)
        except ValueError:
            continue

        decimals_attr = fact.get("decimals")
        scaled_val = raw_val
        try:
            if decimals_attr is not None:
                d = int(decimals_attr)
                if d < 0:
                    # e.g. decimals='-6' and raw='648,125' => 648,125 * 10^6
                    scaled_val = raw_val * (10 ** (-d))
        except Exception:
            # If decimals malformed, just keep raw_val
            scaled_val = raw_val

        # Prefer the largest revenue value if multiple facts for same period
        prev = revenue_by_period.get(period_str)
        if prev is None or scaled_val > prev:
            revenue_by_period[period_str] = float(scaled_val)

    if DEBUG_EXTRACTION_AGENT:
        if revenue_by_period:
            print("[ExtractionAgent] _extract_revenue_direct_from_file: revenue_by_period (from direct iXBRL):")
            for p, v in sorted(revenue_by_period.items()):
                print(f"  period={p}, revenue={v}")
        else:
            print("[ExtractionAgent] _extract_revenue_direct_from_file: no revenue facts found.")

    return revenue_by_period


def _fix_revenue_from_ixbrl(doc_path: str, parsed: Dict[str, Any], facts: Dict[str, Dict[str, float]]) -> None:
    """
    Override facts["revenue"] using true iXBRL revenue facts.

    Strategy:
    1) If parse_10k_html_ixbrl already returned a list of ixbrl facts
       (under common keys), use that.
    2) Otherwise, fall back to directly parsing the HTML file for
       ix:nonFraction revenue facts.
    """

    # --- 1) Try to use ixbrl facts from `parsed` (if present) -----------------
    ixbrl_facts: List[Dict[str, Any]] = (
        parsed.get("ixbrl_facts")
        or parsed.get("raw_ixbrl_facts")
        or parsed.get("facts_list")
        or []
    )

    if ixbrl_facts:
        if DEBUG_EXTRACTION_AGENT:
            print(f"[ExtractionAgent] _fix_revenue_from_ixbrl: #ixbrl_facts={len(ixbrl_facts)} (from parsed dict)")

        REVENUE_CONCEPTS = {
            "us-gaap:Revenues",
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        }

        revenue_by_period: Dict[str, float] = {}

        for fact in ixbrl_facts:
            name = str(fact.get("name") or "").strip()
            if name not in REVENUE_CONCEPTS:
                continue

            period = fact.get("period")
            if isinstance(period, dict):
                period_str = period.get("end") or period.get("instant")
            else:
                period_str = period
            if not period_str:
                continue

            scaled = fact.get("scaled")
            if scaled is None:
                raw = fact.get("raw")
                decimals = fact.get("decimals")
                try:
                    if raw is not None:
                        scaled = float(raw)
                        if isinstance(decimals, int) and decimals < 0:
                            scaled = scaled * (10 ** (-decimals))
                except Exception:
                    continue

            if not isinstance(scaled, (int, float)):
                continue

            prev = revenue_by_period.get(period_str)
            if prev is None or scaled > prev:
                revenue_by_period[period_str] = float(scaled)

        if revenue_by_period:
            if DEBUG_EXTRACTION_AGENT:
                print("[ExtractionAgent] _fix_revenue_from_ixbrl: revenue_by_period (from parsed ixbrl_facts):")
                for p, v in sorted(revenue_by_period.items()):
                    print(f"  period={p}, revenue={v}")

            old_revenue = facts.get("revenue") or {}
            facts["revenue"] = revenue_by_period

            if DEBUG_EXTRACTION_AGENT:
                print("[ExtractionAgent] _fix_revenue_from_ixbrl: overriding facts['revenue']:")
                for p in sorted(revenue_by_period.keys()):
                    old_val = old_revenue.get(p)
                    new_val = revenue_by_period[p]
                    print(f"  {p}: {old_val} -> {new_val}")
            return

        # If we had ixbrl_facts but still no revenue, fall through to fallback.

    else:
        if DEBUG_EXTRACTION_AGENT:
            print("[ExtractionAgent] _fix_revenue_from_ixbrl: no ixbrl_facts list found in parsed; using direct HTML fallback.")

    # --- 2) Fallback: parse iXBRL directly from the HTML file -----------------
    revenue_by_period = _extract_revenue_direct_from_file(doc_path)
    if not revenue_by_period:
        if DEBUG_EXTRACTION_AGENT:
            print("[ExtractionAgent] _fix_revenue_from_ixbrl: fallback also found no revenue; keeping existing facts['revenue'].")
        return

    old_revenue = facts.get("revenue") or {}
    facts["revenue"] = revenue_by_period

    if DEBUG_EXTRACTION_AGENT:
        print("[ExtractionAgent] _fix_revenue_from_ixbrl: overriding facts['revenue'] from direct iXBRL parse:")
        for p in sorted(revenue_by_period.keys()):
            old_val = old_revenue.get(p)
            new_val = revenue_by_period[p]
            print(f"  {p}: {old_val} -> {new_val}")


class ExtractionAgent(BaseAgent):
    """
    Deterministic ExtractionAgent.

    Responsibilities:
    - Read the document path from session.state:
        'html_path' or 'report_path' or 'pdf_path'
    - Call tools.adk_pdf_tools.parse_10k_html_ixbrl(...)
      (which returns scaled numeric facts)
    - Fix the revenue series using the true iXBRL revenue concepts
      (either from parsed ixbrl_facts or by directly parsing the file).
    - Write structured results back into state via EventActions.state_delta:
        'company', 'ticker', 'symbol', 'cik',
        'financials_raw', 'available_periods', 'extraction'
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        # Read current session state (dict)
        state: Dict[str, Any] = ctx.session.state

        if DEBUG_EXTRACTION_AGENT:
            print("\n[ExtractionAgent] START")
            print("[ExtractionAgent] state keys BEFORE:", list(state.keys()))

        # 1) Locate the path in state
        doc_path = (
            state.get("report_path")
            or state.get("html_path")
            or state.get("pdf_path")
        )

        if not doc_path:
            raise ValueError(
                "ExtractionAgent: missing document path in session.state. "
                "Expected one of: 'report_path', 'html_path', or 'pdf_path'."
            )

        if DEBUG_EXTRACTION_AGENT:
            print(f"[ExtractionAgent] INPUT doc_path={doc_path!r}")

        # 2) Call the deterministic parser
        parsed = parse_10k_html_ixbrl(doc_path)

        # Core metadata
        company_name = parsed.get("company_name")
        ticker = parsed.get("ticker")
        # If 'symbol' not explicitly set, fall back to 'ticker'
        symbol = parsed.get("symbol") or ticker
        cik = parsed.get("cik")

        # Financial facts (fully scaled, but revenue may need correction)
        facts = parsed.get("facts") or {}

        # 🔧 2b) Fix revenue series using iXBRL facts if available, else fallback
        _fix_revenue_from_ixbrl(doc_path, parsed, facts)

        # 3) Compute list of all available period end dates
        periods: Set[str] = set()
        for metric_map in facts.values():
            if isinstance(metric_map, dict):
                periods.update(metric_map.keys())
        available_periods = sorted(periods) if periods else []

        # 4) Build state_delta (what we want to persist)
        state_delta: Dict[str, Any] = {
            "extraction": parsed,
            "financials_raw": facts,
        }

        # Company + identifiers
        if company_name is not None:
            state_delta["company"] = company_name
        if ticker is not None:
            state_delta["ticker"] = ticker
        if symbol is not None:
            state_delta["symbol"] = symbol
        if cik is not None:
            state_delta["cik"] = cik

        # Periods
        if available_periods:
            state_delta["available_periods"] = available_periods

        if DEBUG_EXTRACTION_AGENT:
            print("[ExtractionAgent] company_name:", company_name)
            print("[ExtractionAgent] ticker:", ticker)
            print("[ExtractionAgent] symbol:", symbol)
            print("[ExtractionAgent] cik:", cik)
            print("[ExtractionAgent] metrics:", list(facts.keys()))
            print("[ExtractionAgent] available_periods:", available_periods)

            # Extra debug: show sample *corrected* revenue / net_income if available
            revenue = facts.get("revenue") or {}
            net_income = facts.get("net_income") or {}
            if revenue:
                sample_periods = sorted(revenue.keys())
                print("[ExtractionAgent] Sample *corrected* revenue values:")
                for p in sample_periods[:3]:
                    print(f"  period={p}, revenue={revenue[p]}")
            if net_income:
                sample_periods = sorted(net_income.keys())
                print("[ExtractionAgent] Sample scaled net_income values:")
                for p in sample_periods[:3]:
                    print(f"  period={p}, net_income={net_income[p]}")

            print("[ExtractionAgent] state_delta keys:", list(state_delta.keys()))
            print("[ExtractionAgent] END\n")

        # 5) Yield Event with state_delta so SessionService persists it
        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )


extraction_agent = ExtractionAgent(
    name="extraction_agent",
    description=(
        "Deterministic 10-K HTML/iXBRL extractor. "
        "Reads document path from state and writes structured, "
        "FULLY SCALED financial facts + company metadata "
        "(company, ticker, symbol, cik) back into state, with "
        "revenue corrected from iXBRL top-line facts."
    ),
)
