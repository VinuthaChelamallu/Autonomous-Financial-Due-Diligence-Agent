# agents/mas_root.py

from __future__ import annotations

from google.adk.agents import SequentialAgent

# Import agents
from agents.extraction_agent_adk import extraction_agent
from agents.financial_agent_adk import financial_agent
from agents.risk_mdna_agent_adk import risk_mdna_agent
from agents.memo_agent_adk import memo_agent
from agents.company_agent_adk import company_agent
from agents.live_market_agent_adk import live_market_state_agent
from agents.valuation_agent_adk import valuation_agent

from agents.market_analysis_agent_adk import market_analysis_agent


def build_mas() -> SequentialAgent:

    return SequentialAgent(
        name="mas_root",
        sub_agents=[
            extraction_agent,          # Step 1
            financial_agent,           # Step 2
            risk_mdna_agent,           # Step 3
            company_agent,             # Step 4
            market_analysis_agent,     # Step 5 (web market analysis)
            live_market_state_agent,   # Step 6
            valuation_agent,           # Step 7
            memo_agent,                # Step 8
        ],
    )
