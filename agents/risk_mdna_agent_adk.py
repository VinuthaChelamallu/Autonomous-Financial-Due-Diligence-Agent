from __future__ import annotations
import re
import json
from typing import AsyncGenerator, Dict, Any, List

from google import genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from tools.adk_text_tools import (
    extract_10k_text_sections,
    build_10k_section_map,
)

DEBUG_RISK_AGENT = True

# Gemini setup – expects GOOGLE_API_KEY in environment
_GEMINI_CLIENT = genai.Client()  # api key picked up from env
_GEMINI_MODEL_RISK = "gemini-2.0-flash-lite"
_GEMINI_MODEL_MDNA = "gemini-2.0-flash-lite"


def _safe_json_from_text(text: str) -> Dict[str, Any]:
    """
    Try to parse a JSON object from model output.

    Handles common patterns like:
      ```json
      { ... }
      ```
    and other leading/trailing commentary.

    If parsing still fails, returns {'raw_text': text}.
    """
    raw = text.strip()

    # 1) Strip Markdown code fences like ```json ... ```
    if raw.startswith("```"):
        # Remove leading ``` or ```json
        raw = re.sub(r"^```[a-zA-Z0-9]*\s*", "", raw)
        # Remove trailing ```
        raw = re.sub(r"```$", "", raw.strip())
        raw = raw.strip()

    # 2) Try direct JSON parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 3) Fallback: try to extract the first {...} block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    # 4) Give up: keep raw text for debugging
    return {"raw_text": text}


def _analyze_item_1a_risks(item_1a_text: str) -> Dict[str, Any]:
    """
    Call Gemini to extract structured risks from Item 1A text.
    Returns a dict with key 'risks' (list of objects).
    """
    if not item_1a_text.strip():
        return {"risks": []}

    prompt = """
You are a senior financial risk analyst reading the "Item 1A. Risk Factors"
section of a US public company's Form 10-K.

From the text below, extract at most 5 *distinct* material risks.

For each risk, provide:
- name: A short, human-readable title (1 line).
- category: One of [Regulatory, Operational, Market, Competitive, Supply Chain, Technology, Legal, Other].
- description: 2–4 sentences summarizing the risk in plain English.
- financial_impact: A short phrase on how this could affect revenue, margins, cash flow, or valuation.
- likelihood: One of [Low, Medium, High] based on the language in the filing.

Return ONLY valid JSON with this structure:
{
  "risks": [
    {
      "name": "...",
      "category": "...",
      "description": "...",
      "financial_impact": "...",
      "likelihood": "Low | Medium | High"
    }
  ]
}
Do not include any extra commentary or text outside the JSON.
""".strip()

    resp = _GEMINI_CLIENT.models.generate_content(
        model=_GEMINI_MODEL_RISK,
        contents=[prompt, item_1a_text],
    )

    text = resp.text if hasattr(resp, "text") else str(resp)
    data = _safe_json_from_text(text)
    if "risks" not in data or not isinstance(data["risks"], list):
        data["risks"] = []

    return data


def _analyze_mdna_headwinds_tailwinds(mdna_text: str) -> Dict[str, Any]:
    """
    Call Gemini to extract headwinds and tailwinds from MD&A text.
    Returns a dict with keys 'headwinds' and 'tailwinds'.
    """
    if not mdna_text.strip():
        return {"headwinds": [], "tailwinds": []}

    prompt = """
You are an equity research analyst reading the Management's Discussion and
Analysis (MD&A) section of a US public company's Form 10-K.

From the text below, extract:
- up to 3 "headwinds" (negative factors that could pressure performance)
- up to 3 "tailwinds" (positive factors that could support performance)

For each headwind or tailwind, provide:
- name: short title (1 line)
- description: 1–3 sentences summarizing what management is saying.

Return ONLY valid JSON with this structure:
{
  "headwinds": [
    {"name": "...", "description": "..."}
  ],
  "tailwinds": [
    {"name": "...", "description": "..."}
  ]
}
Do not include any extra commentary or text outside the JSON.
""".strip()

    resp = _GEMINI_CLIENT.models.generate_content(
        model=_GEMINI_MODEL_MDNA,
        contents=[prompt, mdna_text],
    )

    text = resp.text if hasattr(resp, "text") else str(resp)
    data = _safe_json_from_text(text)

    if "headwinds" not in data or not isinstance(data["headwinds"], list):
        data["headwinds"] = []
    if "tailwinds" not in data or not isinstance(data["tailwinds"], list):
        data["tailwinds"] = []

    return data


class RiskMdnaAgent(BaseAgent):
    """
    Risk & MD&A Agent (Gemini-powered).

    Responsibilities:
    - Read document path from state.
    - Build a 10-K section map (Item 1A, 7, 8, etc.) and store it in state
      as 'sections_index' for metadata-aware Q&A.
    - Extract Item 1A & Item 7 text.
    - Call Gemini to:
        * Extract structured risks from Item 1A.
        * Extract headwinds/tailwinds from MD&A.
    - Write into state:
        * 'item_1a_text'
        * 'mdna_text'
        * 'risk_mdna_summary'
        * 'sections_index'
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        state: Dict[str, Any] = ctx.session.state

        if DEBUG_RISK_AGENT:
            print("\n[RiskMdnaAgent] START")
            print("[RiskMdnaAgent] state keys BEFORE:", list(state.keys()))

        doc_path = (
            state.get("report_path")
            or state.get("html_path")
            or state.get("pdf_path")
        )

        if not doc_path:
            raise ValueError(
                "RiskMdnaAgent: missing document path in state. "
                "Expected 'html_path', 'report_path', or 'pdf_path'."
            )

        if DEBUG_RISK_AGENT:
            print(f"[RiskMdnaAgent] INPUT doc_path={doc_path!r}")

        # 1) Build a generic 10-K section map (for future metadata filtering)
        section_map = build_10k_section_map(doc_path)
        sections_index = section_map.get("chunks", [])
        sections_by_id = section_map.get("sections_by_id", {})

        # 2) Extract our two main text buckets from this map
        sections = extract_10k_text_sections(doc_path)
        item_1a_raw = sections.get("item_1a_raw", "")
        item_7_raw = sections.get("item_7_raw", "")

        has_item_1a = bool(item_1a_raw.strip())
        has_mdna = bool(item_7_raw.strip())

        # 3) Call Gemini for structured analysis
        risks_data: Dict[str, Any] = {"risks": []}
        mdna_data: Dict[str, Any] = {"headwinds": [], "tailwinds": []}

        if has_item_1a:
            if DEBUG_RISK_AGENT:
                print("[RiskMdnaAgent] Calling Gemini for Item 1A risks...")
            risks_data = _analyze_item_1a_risks(item_1a_raw)

        if has_mdna:
            if DEBUG_RISK_AGENT:
                print("[RiskMdnaAgent] Calling Gemini for MD&A headwinds/tailwinds...")
            mdna_data = _analyze_mdna_headwinds_tailwinds(item_7_raw)

        # 4) Build unified summary
        top_risks: List[Dict[str, Any]] = risks_data.get("risks", [])
        headwinds: List[Dict[str, Any]] = mdna_data.get("headwinds", [])
        tailwinds: List[Dict[str, Any]] = mdna_data.get("tailwinds", [])

        summary: Dict[str, Any] = {
            "has_item_1a": has_item_1a,
            "has_mdna": has_mdna,
            "top_risks": top_risks,
            "headwinds": headwinds,
            "tailwinds": tailwinds,
            "raw_llm": {
                "item_1a": risks_data,
                "mdna": mdna_data,
            },
        }

        state_delta: Dict[str, Any] = {
            "item_1a_text": item_1a_raw,
            "mdna_text": item_7_raw,
            "risk_mdna_summary": summary,
            "sections_index": sections_index,  # 👈 the "map" of the 10-K
        }

        if DEBUG_RISK_AGENT:
            print("[RiskMdnaAgent] extracted item_1a length:", len(item_1a_raw))
            print("[RiskMdnaAgent] extracted mdna length:", len(item_7_raw))
            print("[RiskMdnaAgent] #top_risks:", len(top_risks))
            print("[RiskMdnaAgent] #headwinds:", len(headwinds))
            print("[RiskMdnaAgent] #tailwinds:", len(tailwinds))
            print("[RiskMdnaAgent] sections_index size:", len(sections_index))
            print("[RiskMdnaAgent] state_delta keys:", list(state_delta.keys()))
            print("[RiskMdnaAgent] END\n")

            print("[RiskMdnaAgent] RAW risks_data:", json.dumps(risks_data, indent=2)[:1000])
            print("[RiskMdnaAgent] RAW mdna_data:", json.dumps(mdna_data, indent=2)[:1000])

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )


risk_mdna_agent = RiskMdnaAgent(
    name="risk_mdna_agent",
    description=(
        "Extracts Item 1A (Risk Factors) and Item 7 (MD&A) text from the 10-K, "
        "builds a section map for metadata-aware Q&A, and uses Gemini to "
        "synthesize structured risks and headwinds/tailwinds."
    ),
)

