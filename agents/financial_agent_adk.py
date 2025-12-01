# agents/financial_agent_adk.py
from __future__ import annotations

from typing import Any, Dict, AsyncGenerator, Optional, List

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

DEBUG_FINANCIAL_AGENT = True


def _safe_growth(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    """
    Compute year-over-year growth: (curr - prev) / |prev|
    Returns None if prev is None or zero, or curr is None.
    """
    if curr is None or prev is None or prev == 0:
        return None
    return (curr - prev) / abs(prev)


class FinancialAgent(BaseAgent):
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        state: Dict[str, Any] = ctx.session.state

        if DEBUG_FINANCIAL_AGENT:
            print("\n[FinancialAgent] START")
            print("[FinancialAgent] state keys BEFORE:", list(state.keys()))

        # Raw multi-period facts from tools/adk_pdf_tools.parse_10k_html_ixbrl
        facts: Dict[str, Dict[str, float]] = state.get("financials_raw") or {}
        periods: List[str] = state.get("available_periods") or []

        if not periods:
            if DEBUG_FINANCIAL_AGENT:
                print("[FinancialAgent] ERROR: 'available_periods' missing or empty")
            raise ValueError("FinancialAgent: 'available_periods' missing or empty")

        # Latest period is our key_period (already used in other agents)
        key_period = periods[-1]

        def _get_for_period(friendly_key: str, period: str) -> Optional[float]:
            per_period = facts.get(friendly_key) or {}
            return per_period.get(period)

        if DEBUG_FINANCIAL_AGENT:
            print(f"[FinancialAgent] periods: {periods}")
            print(f"[FinancialAgent] key_period: {key_period}")

        # ------------------------------------------------------------------
        # 1) Build per-period metrics (including true EBITDA where possible)
        # ------------------------------------------------------------------
        financial_metrics: Dict[str, Dict[str, Any]] = {}
        prev_period: Optional[str] = None

        for period in periods:
            revenue = _get_for_period("revenue", period)
            net_income = _get_for_period("net_income", period)
            ocf = _get_for_period("operating_cash_flow", period)
            capex = _get_for_period("capital_expenditures", period)

            # NEW: operating income and D&A from ixbrl
            operating_income = _get_for_period("operating_income", period)
            d_and_a = _get_for_period("depreciation_amortization", period)

            # ---- EBITDA computation (true where possible) ----
            ebitda: Optional[float] = None
            if operating_income is not None and d_and_a is not None:
                # Preferred: true EBITDA = Operating Income + Depreciation & Amortization
                ebitda = operating_income + d_and_a
            elif operating_income is not None:
                # Fallback: at least use operating income
                ebitda = operating_income
            elif d_and_a is not None and net_income is not None:
                # Secondary fallback: NI + D&A (ignores interest & taxes)
                ebitda = net_income + d_and_a
            else:
                # Last resort: keep earlier behavior and use net income
                ebitda = net_income

            # ---- Free Cash Flow ----
            fcf: Optional[float] = None
            if ocf is not None and capex is not None:
                fcf = ocf - capex

            # ---- Margins ----
            ebitda_margin: Optional[float] = None
            net_margin: Optional[float] = None
            fcf_margin: Optional[float] = None

            if revenue and revenue != 0:
                if ebitda is not None:
                    ebitda_margin = ebitda / revenue
                if net_income is not None:
                    net_margin = net_income / revenue
                if fcf is not None:
                    fcf_margin = fcf / revenue

            metrics: Dict[str, Any] = {
                "revenue": revenue,
                "net_income": net_income,
                "operating_cash_flow": ocf,
                "capital_expenditures": capex,
                "operating_income": operating_income,
                "depreciation_amortization": d_and_a,
                "ebitda": ebitda,
                "ebitda_margin": ebitda_margin,
                "net_margin": net_margin,
                "fcf": fcf,
                "fcf_margin": fcf_margin,
            }

            # ---- YoY growth vs previous period (if available) ----
            if prev_period is not None:
                prev_metrics = financial_metrics.get(prev_period, {})
                metrics["revenue_growth"] = _safe_growth(
                    revenue, prev_metrics.get("revenue")
                )
                metrics["net_income_growth"] = _safe_growth(
                    net_income, prev_metrics.get("net_income")
                )
                metrics["fcf_growth"] = _safe_growth(
                    fcf, prev_metrics.get("fcf")
                )
                metrics["ebitda_growth"] = _safe_growth(
                    ebitda, prev_metrics.get("ebitda")
                )

            financial_metrics[period] = metrics
            prev_period = period

        # ------------------------------------------------------------------
        # 2) Build "latest_metrics" + "yoy_metrics" for memo consumers
        # ------------------------------------------------------------------
        latest_metrics: Dict[str, Any] = financial_metrics.get(key_period, {}) or {}
        yoy_metrics: Dict[str, Any] = {
            k: v for k, v in latest_metrics.items() if k.endswith("_growth")
        }

        if DEBUG_FINANCIAL_AGENT:
            print("[FinancialAgent] built financial_metrics for periods:")
            for p in periods:
                print(f"  - {p}")
            print(f"[FinancialAgent] sample metrics for key_period={key_period}:")
            for k in [
                "revenue",
                "net_income",
                "operating_income",
                "depreciation_amortization",
                "ebitda",
                "ebitda_margin",
                "net_margin",
                "fcf",
                "fcf_margin",
                "revenue_growth",
                "net_income_growth",
                "fcf_growth",
                "ebitda_growth",
            ]:
                print(f"  {k}: {latest_metrics.get(k)}")

        # ------------------------------------------------------------------
        # 3) memo_financials contract: KEEP OLD SHAPE + add new fields
        # ------------------------------------------------------------------
        memo_financials = {
            # legacy consumers
            "latest_metrics": latest_metrics,
            "yoy_metrics": yoy_metrics,
            # richer structure for new agents
            "financials_raw": facts,
            "financial_metrics": financial_metrics,
            "key_period": key_period,
            "company": state.get("company"),
            "ticker": state.get("ticker"),
            "symbol": state.get("symbol"),
        }

        state_delta = {
            # per-period metrics (for your final summary / valuation)
            "financial_metrics": financial_metrics,
            # memo-friendly bundle for MemoAgent & others
            "memo_financials": memo_financials,
            "key_period": key_period,
        }

        if DEBUG_FINANCIAL_AGENT:
            print(
                "[FinancialAgent] memo_financials keys:",
                list(memo_financials.keys()),
            )
            print("[FinancialAgent] END\n")

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )


financial_agent = FinancialAgent(
    name="financial_agent",
    description=(
        "Deterministic financial metrics calculator. "
        "Uses extracted 10-K facts (revenue, net income, operating income, "
        "depreciation & amortization, operating cash flow, capex) to compute "
        "per-period EBITDA, margins, FCF, growth and memo-ready latest metrics."
    ),
)
