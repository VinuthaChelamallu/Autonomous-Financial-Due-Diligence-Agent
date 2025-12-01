# agents/live_market_agent_adk.py
from __future__ import annotations

import os
from typing import AsyncGenerator, Dict, Any

import requests
from datetime import datetime, timezone

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_STOCK_SERVER_PATH = os.path.abspath(
    os.path.join(THIS_DIR, "..", "tools", "mcp_live_stock_server.py")
)

# You already use Finnhub in the MCP server; we reuse the same API here
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or "d4lqcfhr01qr851q163gd4lqcfhr01qr851q1640"

DEBUG_LIVE_MARKET_STATE_AGENT = True


# ---------------------------------------------------------------------------
# 1) MAS-FRIENDLY AGENT: writes live_stock into session state
# ---------------------------------------------------------------------------

class LiveMarketStateAgent(BaseAgent):
    """
    Deterministic live market agent for the MAS pipeline.

    Responsibilities:
    - Read the ticker/symbol from session state (set by ExtractionAgent/FinancialAgent).
    - Call Finnhub directly to get a live quote.
    - Write the quote into:
        state["live_stock"] and state["live_quote"]
      so MemoAgent and Investment Thesis can use current_price, etc.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        state: Dict[str, Any] = ctx.session.state

        if DEBUG_LIVE_MARKET_STATE_AGENT:
            print("\n[LiveMarketStateAgent] START")
            print("[LiveMarketStateAgent] state keys BEFORE:", list(state.keys()))

        symbol = (
            state.get("ticker")
            or state.get("symbol")
            or state.get("company_ticker")
        )

        if not symbol:
            if DEBUG_LIVE_MARKET_STATE_AGENT:
                print("[LiveMarketStateAgent] No ticker/symbol found in state. Skipping live quote.\n")
            # no-op
            yield Event(author=self.name, actions=EventActions())
            return

        if not FINNHUB_API_KEY:
            if DEBUG_LIVE_MARKET_STATE_AGENT:
                print("[LiveMarketStateAgent] FINNHUB_API_KEY not set. Skipping live quote.\n")
            yield Event(author=self.name, actions=EventActions())
            return

        # Call Finnhub (same as your MCP tool, but directly here)
        url = "https://finnhub.io/api/v1/quote"
        params = {"symbol": symbol, "token": FINNHUB_API_KEY}

        quote: Dict[str, Any] = {
            "symbol": symbol,
            "current_price": None,
            "high_price": None,
            "low_price": None,
            "open_price": None,
            "previous_close_price": None,
            "change": None,
            "change_percent": None,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            current = data.get("c")
            prev_close = data.get("pc")
            change = None
            change_pct = None

            if current is not None and prev_close not in (None, 0):
                change = current - prev_close
                change_pct = (change / prev_close) * 100.0

            quote.update(
                {
                    "current_price": current,
                    "high_price": data.get("h"),
                    "low_price": data.get("l"),
                    "open_price": data.get("o"),
                    "previous_close_price": prev_close,
                    "change": change,
                    "change_percent": change_pct,
                }
            )

            if DEBUG_LIVE_MARKET_STATE_AGENT:
                print("[LiveMarketStateAgent] Fetched live quote from Finnhub:")
                print(quote)

        except Exception as e:
            if DEBUG_LIVE_MARKET_STATE_AGENT:
                print(f"[LiveMarketStateAgent] Error fetching live quote for {symbol}: {e}")

        state_delta = {
            "live_stock": quote,
            "live_quote": quote,  # for backward compatibility
        }

        if DEBUG_LIVE_MARKET_STATE_AGENT:
            print("[LiveMarketStateAgent] END, state_delta keys:", list(state_delta.keys()))
            print()

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )


# This is what you will import into mas_root.py
live_market_state_agent = LiveMarketStateAgent(
    name="live_market_state_agent",
    description=(
        "Fetches live stock quote from Finnhub using the ticker in state, "
        "and writes it to state['live_stock'] for use by MemoAgent."
    ),
)


# ---------------------------------------------------------------------------
# 2) ORIGINAL MCP-BASED LLM AGENT (for manual/debug use outside MAS)
# ---------------------------------------------------------------------------

def build_live_market_agent():
    """
    Build an LLM agent wired to the Finnhub MCP server.
    This is mainly for manual tests / chat-style usage (not MAS state wiring).
    """
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python",
                args=[MCP_STOCK_SERVER_PATH],
            )
        )
    )

    agent = LlmAgent(
        model="gemini-2.0-flash-lite",
        name="live_market_agent",
        instruction=(
            "You are a stock assistant. "
            "Always use the 'live_stock_quote' MCP tool to fetch real-time "
            "prices and then respond with clean JSON."
        ),
        tools=[toolset],
    )

    return agent, toolset
