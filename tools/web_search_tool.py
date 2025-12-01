# tools/web_search_tool.py
from __future__ import annotations

from google.adk.tools.google_search_tool import GoogleSearchTool


def build_web_search_tool() -> GoogleSearchTool:
    """
    Factory for the ADK GoogleSearchTool.

    This tool lets the LLM perform live Google web search to fetch:
      - market size (TAM/SAM/SOM)
      - growth rates (CAGR)
      - competitors and market share
      - recent industry trends

    Intended to be used by the MarketAnalysisAgent.
    """
    # Your ADK version does NOT accept 'name=' in the ctor.
    # Keep it simple and rely on default settings.
    return GoogleSearchTool(bypass_multi_tools_limit=True)
