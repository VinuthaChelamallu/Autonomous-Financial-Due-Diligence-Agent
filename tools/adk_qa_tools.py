"""
Routing and context-prep utilities for the Q&A agent.

Goal:
  Given a natural language question and the current session.state,
  decide *which* bucket(s) of data to use:
    - financial metrics (numbers)
    - Item 1A Risk Factors (risks)
    - MD&A Item 7 (management discussion)
    - combinations of the above (multi-intent)
    - generic / fallback

This module is primarily deterministic, but has an OPTIONAL LLM
fallback classifier when no intent is detected (hybrid routing).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from google import genai


@dataclass
class QaRouteResult:
    """
    Structured routing decision for a single user question.

    route:
      - "financials"                    -> mainly numbers
      - "risks"                         -> mainly Item 1A
      - "mdna"                          -> mainly Item 7
      - "auto_multi:financials,risks"   -> combined, section-aware
      - "generic"                       -> generic/fallback

    evidence:
      - A dict with the minimal (but structured) slice of state that
        the Q&A agent should see.
      - Always includes:
          - "company"
          - "question"
          - "intents": list[str]
          - "sections_index"
          - "sections_index_filtered"
          - "financials"
          - "risks"
          - "mdna"
    """

    route: str
    question: str
    evidence: Dict[str, Any]


# ---------------------------
# Normalization & intent heuristics (deterministic)
# ---------------------------

def _normalize(text: str) -> str:
    return (text or "").lower()


def _looks_like_risk_question(q: str) -> bool:
    qn = _normalize(q)
    keywords = [
        "risk",
        "risks",
        "risk factor",
        "risk factors",
        "what are the main risks",
        "key risks",
        "risk profile",
    ]
    return any(kw in qn for kw in keywords)


def _looks_like_mdna_question(q: str) -> bool:
    qn = _normalize(q)
    keywords = [
        "md&a",
        "mdna",
        "management's discussion",
        "managements discussion",
        "management discussion",
        "results of operations",
        "outlook",
        "future outlook",
        "what does management say",
    ]
    return any(kw in qn for kw in keywords)


def _looks_like_financials_question(q: str) -> bool:
    qn = _normalize(q)
    metric_words = [
        "revenue",
        "sales",
        "net income",
        "profit",
        "earnings",
        "free cash flow",
        "fcf",
        "cash flow",
        "operating cash",
        "operating cash flow",
        "margin",
        "gross margin",
        "operating margin",
        "net margin",
        "ebitda",  # we may not compute it yet, but it's clearly financial
        "growth",
        "year over year",
        "yoy",
    ]
    return any(kw in qn for kw in metric_words)


# ---------------------------
# Evidence builders
# ---------------------------

def _extract_financial_evidence(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull the minimum financial slice from state for answering
    numeric/metric questions.
    """
    financial_metrics = state.get("financial_metrics") or {}
    key_period = state.get("key_period")
    periods = financial_metrics.get("periods") or []
    per_period = financial_metrics.get("per_period") or {}
    yoy = financial_metrics.get("yoy") or {}

    metrics_for_period = {}
    if key_period and key_period in per_period:
        metrics_for_period = per_period.get(key_period, {})

    return {
        "company": state.get("company"),
        "key_period": key_period,
        "periods": periods,
        "metrics_for_period": metrics_for_period,
        "yoy": yoy,
    }


def _extract_risks_evidence(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull the minimum risk slice: Item 1A text and structured risk summary.
    """
    risk_summary = state.get("risk_mdna_summary") or {}
    item_1a_text = state.get("item_1a_text") or ""

    return {
        "company": state.get("company"),
        "item_1a_text": item_1a_text,
        "risk_summary": {
            "top_risks": risk_summary.get("top_risks", []),
            "has_item_1a": risk_summary.get("has_item_1a"),
        },
    }


def _extract_mdna_evidence(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull the minimum MD&A slice: Item 7 text and headwinds/tailwinds.
    """
    risk_summary = state.get("risk_mdna_summary") or {}
    mdna_text = state.get("mdna_text") or ""

    return {
        "company": state.get("company"),
        "mdna_text": mdna_text,
        "mdna_summary": {
            "headwinds": risk_summary.get("headwinds", []),
            "tailwinds": risk_summary.get("tailwinds", []),
            "has_mdna": risk_summary.get("has_mdna"),
        },
    }


# ---------------------------
# Section-index helpers (10-K map)
# ---------------------------

def _filter_sections_index_for_intents(
    intents: Set[str],
    sections_index: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Given a set of high-level intents {"financials","risks","mdna"},
    slice the sections_index down to only the most relevant sections.

    This doesn't need to be perfect – it's just a hint to the LLM so it
    knows which parts of the 10-K the evidence is coming from.
    """

    if not sections_index:
        return {}

    # Map intent -> which section_type values we care about
    intent_to_types = {
        "financials": {"financials", "market_risk", "equity_market"},
        "risks": {"risks", "cybersecurity", "market_risk", "safety"},
        "mdna": {"mdna", "market_risk", "financials"},
    }

    allowed_types: Set[str] = set()
    for intent in intents:
        allowed_types.update(intent_to_types.get(intent, set()))

    # If we somehow have no allowed types (e.g. no intents),
    # just return an empty filtered index.
    if not allowed_types:
        return {}

    filtered: Dict[str, Any] = {}
    for sec_id, meta in sections_index.items():
        sec_type = (meta or {}).get("section_type")
        if sec_type in allowed_types:
            filtered[sec_id] = meta

    return filtered


# ---------------------------
# LLM-based intent classifier (fallback)
# ---------------------------

_INTENT_CLIENT = genai.Client()
_INTENT_MODEL = "gemini-2.0-flash-lite"  # lightweight, fast; change if needed


def _classify_intents_with_llm(question: str) -> List[str]:
    """
    Use Gemini to classify the question into zero or more of:
      - "financials"
      - "risks"
      - "mdna"

    This is ONLY used as a fallback when the deterministic
    keyword heuristics detect no intents.
    """
    q = (question or "").strip()
    if not q:
        return []

    prompt = f"""
You are an intent classifier for questions about a public company's 10-K filing.

Classify the user's question into zero, one, or more of these labels:
- "financials"  (questions about revenue, profit, margins, cash flow, growth, etc.)
- "risks"       (questions about risk factors, threats, uncertainties, downside)
- "mdna"        (questions about management's discussion, headwinds, tailwinds, outlook)

Rules:
- Return ONLY a JSON list of strings, like:
  ["financials"]
  ["financials","risks"]
  ["mdna","risks"]
- If nothing matches, return [].
- Do NOT include any explanation or text outside the JSON list.

User question:
{q}
""".strip()

    try:
        resp = _INTENT_CLIENT.models.generate_content(
            model=_INTENT_MODEL,
            contents=prompt,
        )
        raw = getattr(resp, "text", "") or str(resp)

        raw = raw.strip()

        # Simple, robust JSON extraction: if it's not valid JSON, fall back to []
        try:
            intents = json.loads(raw)
            if isinstance(intents, list):
                # normalize and filter valid labels
                cleaned = []
                for x in intents:
                    if not isinstance(x, str):
                        continue
                    xl = x.strip().lower()
                    if xl in {"financials", "risks", "mdna"}:
                        cleaned.append(xl)
                return cleaned
        except Exception:
            # If parsing fails, just ignore and return no intents.
            return []
    except Exception:
        # If LLM call fails for any reason (API key, network, etc.), fall back to no intents.
        return []

    return []


# ---------------------------
# Main routing function (hybrid: deterministic + LLM fallback)
# ---------------------------

def route_question(question: str, state: Dict[str, Any]) -> QaRouteResult:
    """
    Main entry point: decide which bucket(s) to use for this question
    and return a structured routing decision + evidence.

    Logic:
      1. Run deterministic keyword heuristics to detect intents.
      2. If no intents are found, call a small LLM classifier to
         infer intents ("financials","risks","mdna").
      3. Build a combined evidence object with:
         - financials, risks, mdna slices
         - 10-K section index + filtered subset
         - the detected intents
      4. Decide route:
         - "financials" / "risks" / "mdna" for single-intent questions
         - "auto_multi:financials,risks" for multi-intent questions
         - "generic" if nothing matched clearly
    """
    q = question or ""

    # -------------------
    # 1) Deterministic detection
    # -------------------
    is_fin = _looks_like_financials_question(q)
    is_risk = _looks_like_risk_question(q)
    is_mdna = _looks_like_mdna_question(q)

    intents: Set[str] = set()
    if is_fin:
        intents.add("financials")
    if is_risk:
        intents.add("risks")
    if is_mdna:
        intents.add("mdna")

    # -------------------
    # 2) LLM fallback classification (only if no intents)
    # -------------------
    if not intents:
        llm_intents = _classify_intents_with_llm(q)
        for label in llm_intents:
            if label == "financials":
                intents.add("financials")
            elif label == "risks":
                intents.add("risks")
            elif label == "mdna":
                intents.add("mdna")

    # -------------------
    # 3) Build evidence (shared for all routes)
    # -------------------
    sections_index = state.get("sections_index") or {}
    sections_index_filtered = _filter_sections_index_for_intents(intents, sections_index)

    financial_evidence = _extract_financial_evidence(state)
    risks_evidence = _extract_risks_evidence(state)
    mdna_evidence = _extract_mdna_evidence(state)

    evidence: Dict[str, Any] = {
        "company": state.get("company"),
        "question": q,
        "intents": sorted(list(intents)),
        "sections_index": sections_index,
        "sections_index_filtered": sections_index_filtered,
        "financials": financial_evidence,
        "risks": risks_evidence,
        "mdna": mdna_evidence,
    }

    # -------------------
    # 4) Decide route label
    # -------------------
    if not intents:
        route = "generic"
    elif len(intents) == 1:
        route = next(iter(intents))  # "financials" or "risks" or "mdna"
    else:
        # deterministic multi-intent ordering
        route = "auto_multi:" + ",".join(sorted(intents))

    return QaRouteResult(
        route=route,
        question=q,
        evidence=evidence,
    )
