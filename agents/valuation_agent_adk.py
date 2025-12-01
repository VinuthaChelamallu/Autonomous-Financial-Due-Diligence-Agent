# agents/valuation_agent_adk.py
from __future__ import annotations

from typing import Any, Dict, AsyncGenerator, Optional, List

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

DEBUG_VALUATION_AGENT = True


class ValuationAgent(BaseAgent):
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        state: Dict[str, Any] = ctx.session.state

        if DEBUG_VALUATION_AGENT:
            print("\n[ValuationAgent] START")
            print("[ValuationAgent] state keys BEFORE:", list(state.keys()))

        # 1) Key period
        periods: List[str] = state.get("available_periods") or []
        key_period: Optional[str] = state.get("key_period")

        if not key_period:
            if periods:
                key_period = periods[-1]
            else:
                if DEBUG_VALUATION_AGENT:
                    print("[ValuationAgent] ERROR: no key_period and no available_periods")
                yield Event(
                    author=self.name,
                    actions=EventActions(
                        state_delta={
                            "valuation_metrics": {
                                "current_price": None,
                                "revenue": None,
                                "shares_raw": None,
                                "shares_source": None,
                                "shares": None,
                                "market_cap": None,
                                "market_cap_millions": None,
                                "pe": None,
                                "fcf_yield": None,
                                "fcf_yield_pct": None,
                                "key_period": None,
                            }
                        }
                    ),
                )
                if DEBUG_VALUATION_AGENT:
                    print("[ValuationAgent] END (no periods)\n")
                return

        if DEBUG_VALUATION_AGENT:
            print(f"[ValuationAgent] key_period={key_period}")

        # 2) Get deterministic financial metrics for that period
        financial_metrics: Dict[str, Dict[str, Any]] = state.get("financial_metrics") or {}
        metrics: Dict[str, Any] = financial_metrics.get(key_period, {}) or {}

        # Fallback: older / alternative shape via memo_financials
        if not metrics:
            memo_financials = state.get("memo_financials") or {}
            metrics = memo_financials.get("latest_metrics") or {}

        revenue: Optional[float] = metrics.get("revenue")
        net_income: Optional[float] = metrics.get("net_income")
        fcf: Optional[float] = metrics.get("fcf")

        if DEBUG_VALUATION_AGENT:
            print(f"[ValuationAgent] revenue={revenue}")
            print(f"[ValuationAgent] net_income={net_income}")
            print(f"[ValuationAgent] fcf={fcf}")

        # 3) Live quote from Finnhub via MCP
        live_quote: Dict[str, Any] = state.get("live_quote") or state.get("live_stock") or {}
        current_price: Optional[float] = live_quote.get("current_price")

        if DEBUG_VALUATION_AGENT:
            print(f"[ValuationAgent] current_price={current_price}")

        # 4) Shares outstanding – prefer deterministic iXBRL shares
        facts: Dict[str, Dict[str, float]] = state.get("financials_raw") or {}
        shares_raw: Optional[float] = None
        shares_source: Optional[str] = None

        # Try in order of "niceness"
        share_keys_order = [
            "common_shares_outstanding",
            "shares_outstanding",
            "weighted_avg_shares_diluted",
            "weighted_avg_shares_basic",
        ]

        for key in share_keys_order:
            per_period = facts.get(key) or {}
            val = per_period.get(key_period)
            if val is not None:
                shares_raw = val
                shares_source = key
                break

        shares = shares_raw

        if DEBUG_VALUATION_AGENT:
            print(f"[ValuationAgent] shares_raw={shares_raw} (source={shares_source})")

        # 5) Market cap & ratios
        market_cap: Optional[float] = None
        pe: Optional[float] = None
        fcf_yield: Optional[float] = None
        fcf_yield_pct: Optional[str] = None

        if current_price is not None and shares is not None:
            market_cap = current_price * shares

            if net_income:
                pe = market_cap / net_income

            if fcf and market_cap:
                fcf_yield = fcf / market_cap
                fcf_yield_pct = f"{fcf_yield * 100.0:.2f}%"

        market_cap_millions: Optional[float] = None
        if market_cap is not None:
            market_cap_millions = market_cap / 1_000_000.0

        valuation_metrics = {
            "current_price": current_price,
            "revenue": revenue,
            "shares_raw": shares_raw,
            "shares_source": shares_source,
            "shares": shares,
            "market_cap": market_cap,
            "market_cap_millions": market_cap_millions,
            "pe": pe,
            "fcf_yield": fcf_yield,
            "fcf_yield_pct": fcf_yield_pct,
            "key_period": key_period,
        }

        if DEBUG_VALUATION_AGENT:
            print("[ValuationAgent] valuation_metrics computed:")
            print(valuation_metrics)
            print("[ValuationAgent] END\n")

        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={"valuation_metrics": valuation_metrics}
            ),
        )


valuation_agent = ValuationAgent(
    name="valuation_agent",
    description=(
        "Deterministic valuation metrics calculator. "
        "Uses financial_metrics for the key period plus live_quote and "
        "share count from iXBRL to compute market cap, P/E and FCF yield."
    ),
)
