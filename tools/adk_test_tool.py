import os
import asyncio
from dotenv import load_dotenv

from google.adk.runners import InMemoryRunner
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search  # built-in ADK google_search tool


# --------------------
# LOAD GOOGLE API KEY
# --------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found. Add it to .env or set it in PowerShell.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# --------------------
# DEFINE TEST AGENT
# --------------------
google_search_test_agent = Agent(
    name="GoogleSearchTestAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",   # ✅ no retry_options here
    ),
    instruction="""
You are a tool-using agent.

You MUST call the `google_search` tool.
Use this exact query: "Apple Inc market size".

After the tool call returns:
- Extract the top few results
- Summarize them in 3–5 bullet points

You are not allowed to answer from your own knowledge.
You MUST use the tool.
""",
    tools=[google_search],
    output_key="search_result",
)

runner = InMemoryRunner(agent=google_search_test_agent)


# --------------------
# ASYNC ENTRYPOINT
# --------------------
async def main():
    print("\n=== RUNNING GOOGLE SEARCH TOOL TEST (LOCAL) ===\n")

    response = await runner.run_debug("Run google search test now")

    print("\n=== GOOGLE SEARCH TOOL TEST COMPLETE ===")
    print("\nFinal result:\n", response)


# --------------------
# RUN
# --------------------
if __name__ == "__main__":
    asyncio.run(main())
