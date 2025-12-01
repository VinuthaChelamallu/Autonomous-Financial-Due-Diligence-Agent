from __future__ import annotations

import os
import requests
from mcp.server.fastmcp import FastMCP

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or "put your key"
if not FINNHUB_API_KEY:
    raise RuntimeError("FINNHUB_API_KEY not set")

mcp = FastMCP("LiveStockServer", json_response=True)


@mcp.tool()
def live_stock_quote(symbol: str = "AAPL") -> dict:
    """
    Get live stock data for any ticker using Finnhub.
    Returns:
      {
        "symbol",
        "current_price",
        "high_price",
        "low_price",
        "open_price",
        "previous_close_price",
        "change",
        "change_percent"
      }
    """
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol, "token": FINNHUB_API_KEY}
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

    return {
        "symbol": symbol,
        "current_price": current,
        "high_price": data.get("h"),
        "low_price": data.get("l"),
        "open_price": data.get("o"),
        "previous_close_price": prev_close,
        "change": change,
        "change_percent": change_pct,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
