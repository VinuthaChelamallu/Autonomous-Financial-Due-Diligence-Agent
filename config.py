# config.py
# Central config for paths used across agents

import os

BASE_DIR = "/workspaces/Finance-Agent/capstone_agent"

MCP_STOCK_SERVER_PATH = os.path.join(
    BASE_DIR,
    "tools",
    "mcp_live_stock_server.py"
)

DEFAULT_COMPANY = "Apple Inc."
