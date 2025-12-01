from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, Set, List, Optional, Tuple

from google import genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from agents.hallucination_guard import (
    validate_money_values,
    append_validation_summary,
)

DEBUG_QA_AGENT = True

_GEMINI_CLIENT = genai.Client()
_GEMINI_MODEL_QA = "gemini-2.0-flash-lite"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _compact_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# Simple detector for money-like numeric values in the answer
_MONEYISH_PATTERN = re.compile(
    r"(\$?\s*\d+(?:\.\d+)?\s*(billion|bn|million|m|thousand|k|%)?)",
    re.IGNORECASE,
)


def _has_money_like_numbers(text: str) -> bool:
    """
    Return True if the text appears to contain money-like or percentage
    numbers (e.g., $118 billion, 12.5%, 391m).
    """
    if not text:
        return False
    return bool(_MONEYISH_PATTERN.search(text))


def _update_conversation_summary(
    existing_summary: str,
    qa_pairs: List[Dict[str, str]],
) -> str:
    """
    Rolling conversation summary à la Day-3b:

    - Takes the existing summary (may be empty).
    - Takes a list of recent Q&A pairs (we pass in history + this turn).
    - Returns an updated compact summary capturing:
        * What has already been discussed about this filing.
        * Key financial points, risks, and narrative themes.
    """
    # Nothing new to summarize
    if not qa_pairs and not existing_summary:
        return ""

    # Build a readable block of Q&A text
    qa_text_blocks: List[str] = []
    for idx, pair in enumerate(qa_pairs, start=1):
        q = pair.get("q") or ""
        a = pair.get("a") or ""
        qa_text_blocks.append(f"Q{idx}: {q}\nA{idx}: {a}")

    qa_text = "\n\n".join(qa_text_blocks)

    prompt = f"""
You are maintaining a rolling summary of a Q&A conversation
about a company's Form 10-K (financial performance, risks,
MD&A, and overall assessment).

You are given:
1) The existing summary (may be empty).
2) The latest conversation snippets (Q&A pairs).

Your task:
- Produce an UPDATED, COMPACT SUMMARY that a future assistant
  could read to quickly understand what has already been covered.
- Keep the summary under ~250 words.
- Focus on:
  * Key financial metrics and trends.
  * Main risks and headwinds/tailwinds.
  * Important management or strategic points.
  * Any conclusions or caveats already discussed.
- You do NOT need to mention each individual question explicitly.
- Do NOT add any new facts; only compress what is in the inputs.

Return ONLY the updated summary text, with no headings or metadata.

--- EXISTING SUMMARY ---
{existing_summary or "(none yet)"}

--- NEW Q&A PAIRS ---
{qa_text}
""".strip()

    try:
        resp = _GEMINI_CLIENT.models.generate_content(
            model=_GEMINI_MODEL_QA,
            contents=[prompt],
        )
        text = getattr(resp, "text", None)
        if not text:
            return existing_summary
        return text.strip()
    except Exception as e:
        if DEBUG_QA_AGENT:
            print(f"[QaAgent] conversation_summary update error: {e}")
        # On any failure, just keep the old summary
        return existing_summary


def _format_live_quote_answer(
    live_quote: Dict[str, Any],
    company: Optional[str] = None,
) -> str:
    """
    Turn the live_quote dict from session.state into a nice human answer.
    Assumes keys: symbol, as_of, current_price, high_price, low_price,
    open_price, previous_close_price, change, change_percent.
    """
    symbol = live_quote.get("symbol") or "N/A"
    as_of = live_quote.get("as_of")
    current = live_quote.get("current_price")
    high = live_quote.get("high_price")
    low = live_quote.get("low_price")
    open_ = live_quote.get("open_price")
    prev_close = live_quote.get("previous_close_price")
    change = live_quote.get("change")
    change_pct = live_quote.get("change_percent")

    # Pretty timestamp
    as_of_str = ""
    if as_of:
        try:
            dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            as_of_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            as_of_str = as_of

    name = company or f"{symbol}"

    lines: List[str] = []

    if isinstance(current, (int, float)):
        lines.append(
            f"The current stock price for **{name} ({symbol})** is **${current:.2f}**."
        )
    else:
        lines.append(
            f"The current stock price for **{name} ({symbol})** is not available."
        )

    if isinstance(change, (int, float)) and isinstance(change_pct, (int, float)):
        sign = "+" if change >= 0 else ""
        lines.append(
            f"That is a change of **{sign}{change:.2f}** ({sign}{change_pct:.2f}%)."
        )

    # Extra detail block
    details: List[str] = []
    if isinstance(open_, (int, float)):
        details.append(f"- Open: ${open_:.2f}")
    if isinstance(prev_close, (int, float)):
        details.append(f"- Previous close: ${prev_close:.2f}")
    if isinstance(high, (int, float)):
        details.append(f"- Day high: ${high:.2f}")
    if isinstance(low, (int, float)):
        details.append(f"- Day low: ${low:.2f}")

    if details:
        lines.append("\n**Intraday details:**")
        lines.extend(details)

    if as_of_str:
        lines.append(f"\n_As of: {as_of_str}_")

    return "\n".join(lines)


# ---------------- Live-market detection (for guidance) --------------------

def _wants_live_quote(question: str) -> bool:
    """
    Heuristic: does the user seem to be asking about stock price / live market?

    We keep this purely keyword-based so it's deterministic.
    """
    q = (question or "").lower()
    keywords = [
        "stock price",
        "share price",
        "market price",
        "today’s price",
        "today's price",
        "todays price",
        "live price",
        "live market",
        "market cap",
        "market capitalization",
        "valuation today",
        "current price",
        "trading at",
        "stock is trading",
        "share is trading",
        "current market cap",
    ]
    return any(kw in q for kw in keywords)


# ---------------- Intent Detection (Hybrid: rules + LLM) -------------------

def _detect_intents_rule_based(question: str) -> Set[str]:
    """
    Deterministic intent detection based on keywords.

    Returns a set of labels among:
      - "financials"
      - "risks"
      - "mdna"
      - "memo"
    """
    q = (question or "").lower()

    intents: Set[str] = set()

    # Financial-ish words
    if any(
        word in q
        for word in [
            "financial highlight",
            "financial highlights",
            "financials",
            "results",
            "performance",
            "earnings",
            "revenue",
            "sales",
            "net income",
            "profit",
            "margin",
            "cash flow",
            "operating cash",
            "free cash flow",
            "fcf",
        ]
    ):
        intents.add("financials")

    # Risk-ish words
    if any(
        word in q
        for word in [
            "risk",
            "risks",
            "risk factor",
            "risk factors",
            "headwind",
            "headwinds",
            "tailwind",
            "tailwinds",
            "downside",
            "threat",
            "threats",
            "uncertainty",
            "volatility",
        ]
    ):
        intents.add("risks")

    # MD&A-ish words
    if any(
        word in q
        for word in [
            "md&a",
            "mdna",
            "management’s discussion",
            "management's discussion",
            "management discussion",
            "item 7",
            "item seven",
        ]
    ):
        intents.add("mdna")

    # Memo-ish / overall questions
    if any(
        word in q
        for word in [
            "investment memo",
            "investment thesis",
            "overall view",
            "overall assessment",
            "due diligence summary",
            "high level",
            "summary view",
        ]
    ):
        intents.add("memo")

    # If we detected nothing specific, default to "memo"/overall style
    if not intents:
        intents.add("memo")

    return intents


def _refine_intents_with_llm(question: str, rule_intents: Set[str]) -> Set[str]:
    """
    Hybrid upgrade: use Gemini to clean up / correct intents,
    especially when user wording is noisy or misspelled.

    We *never* fully trust the LLM here; we only adjust within the
    allowed set and always fall back to rule-based if parsing fails.
    """
    allowed = {"financials", "risks", "mdna", "memo"}

    prompt = f"""
You are an intent classifier for questions about a company's 10-K.

Allowed intent labels:
- "financials"  (metrics, revenue, profit, cash flow, margins, growth)
- "risks"       (risk factors, headwinds, threats, uncertainties)
- "mdna"        (MD&A, Item 7, management's discussion)
- "memo"        (overall assessment, investment memo style summary)

User question:
{question}

Task:
1. Decide which of the above intents are relevant for answering the question.
2. Return ONLY a JSON object of the form:
   {{"intents": ["financials", "risks"]}}

Rules:
- Use only the four allowed labels.
- You can return 1–3 intents.
- Do NOT add any extra fields or text outside the JSON.
""".strip()

    try:
        resp = _GEMINI_CLIENT.models.generate_content(
            model=_GEMINI_MODEL_QA,
            contents=[prompt],
        )
        text = getattr(resp, "text", "") or str(resp)
        data = json.loads(text)
        llm_intents_raw = data.get("intents", [])
        llm_intents = {i for i in llm_intents_raw if i in allowed}

        # Combine rule-based + LLM, so we never get *less* than rules.
        combined = set(rule_intents) | llm_intents

        if not combined:
            # Extreme fallback – should rarely happen
            return rule_intents or {"memo"}

        return combined
    except Exception:
        # If anything goes wrong (JSON parse etc), trust rules
        return rule_intents


def _detect_intents(question: str) -> Set[str]:
    """
    Public entrypoint for intent detection:
      1) run deterministic rules
      2) let LLM refine / correct them
    """
    rule_intents = _detect_intents_rule_based(question)
    refined = _refine_intents_with_llm(question, rule_intents)
    return refined


# ---------------- Section Index Normalization -----------------------------

def _normalize_sections_index(sections_index_raw: Any) -> Dict[str, Any]:
    """
    Normalize sections_index from state into a dict keyed by section id.

    Supports:
      - dict form: {"item_1a": {...}, "item_7": {...}}
      - list form: [{"id": "item_1a", ...}, {"id": "item_7", ...}, ...]
    """
    if not sections_index_raw:
        return {}

    # Newer form: already a dict
    if isinstance(sections_index_raw, dict):
        return sections_index_raw

    # Older form: list of section dicts
    if isinstance(sections_index_raw, list):
        out: Dict[str, Any] = {}
        for entry in sections_index_raw:
            if not isinstance(entry, dict):
                continue
            sid = entry.get("id")
            if not sid:
                continue
            out[sid] = entry
        return out

    # Anything else – ignore
    return {}


# ---------------- Section Index Filtering (10-K map) ----------------------

def _filter_sections_index_by_intents(
    sections_index_raw: Any,
    intents: Set[str],
) -> Dict[str, Any]:
    """
    Take whatever is in state["sections_index"] (dict or list),
    normalize it, and keep only the sections relevant to the detected intents.

    Mapping:
      - "risks"      -> section_type == "risks"
      - "mdna"       -> section_type == "mdna"
      - "financials" -> section_type == "financials"

    If no relevant intents are present, return the full normalized index.
    """
    index = _normalize_sections_index(sections_index_raw)
    if not index:
        return {}

    # Which section_type values do we care about for this question?
    wanted_types: Set[str] = set()
    if "risks" in intents:
        wanted_types.add("risks")
    if "mdna" in intents:
        wanted_types.add("mdna")
    if "financials" in intents:
        wanted_types.add("financials")

    # If nothing specific is requested, just return everything
    if not wanted_types:
        return index

    filtered: Dict[str, Any] = {}
    for sid, meta in index.items():
        stype = (meta or {}).get("section_type")
        if stype in wanted_types:
            filtered[sid] = meta

    return filtered


# ---------------- Fallback Retrieval over sections_index ------------------

def _normalize_text(text: str) -> List[str]:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def _score_section_for_question(question: str, section: Dict[str, Any]) -> float:
    """
    Simple lexical overlap score between the question and the section's
    title + a bit of its text.
    """
    q_tokens = set(_normalize_text(question))

    title = section.get("label") or section.get("title") or ""
    text = section.get("text", "") or ""

    # Only use first N chars of text to keep it cheap
    text_snippet = text[:2000]

    s_tokens = set(_normalize_text(title + " " + text_snippet))

    if not s_tokens:
        return 0.0

    overlap = q_tokens.intersection(s_tokens)
    # simple Jaccard-like-ish score
    score = len(overlap) / max(len(q_tokens), 1)
    return score


def _fallback_sections_for_question(
    question: str,
    sections_index_norm: Dict[str, Any],
    top_k: int = 5,
    min_score: float = 0.1,
) -> Dict[str, Any]:
    """
    When sections_index_filtered is empty, use this to pick the top_k most
    relevant sections for the question based on lexical overlap.

    Returns a dict {section_id -> meta} in the same shape as the normalized index.
    """
    scored: List[Tuple[float, str]] = []

    for sid, meta in sections_index_norm.items():
        if not isinstance(meta, dict):
            continue
        score = _score_section_for_question(question, meta)
        if score > 0:
            scored.append((score, sid))

    # sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    results: Dict[str, Any] = {}
    for score, sid in scored:
        if score < min_score:
            break
        results[sid] = sections_index_norm[sid]
        if len(results) >= top_k:
            break

    return results


# ---------------- Citations Builder --------------------------------------

def _build_citations_text(
    company: str,
    key_period: str,
    intents: Set[str],
    sections_index_filtered: Dict[str, Any],
) -> str:
    """
    Build a human-readable list of sources used for this answer,
    based on intents and which sections were actually selected.
    """
    lines: List[str] = []

    # High-level company / period
    base_company = company or "the company"
    base_period = key_period or "the latest reported period"

    # Intent-based sources
    if "financials" in intents:
        lines.append(
            f"- Financial metrics for {base_period}: extracted from structured facts parsed from "
            f"{base_company} Form 10-K (Item 8, Financial Statements and Supplementary Data)."
        )

    if "risks" in intents:
        lines.append(
            "- Risks and headwinds: derived from Item 1A (Risk Factors) and related discussion in "
            "Item 7 (Management’s Discussion and Analysis) of the Form 10-K."
        )

    if "mdna" in intents:
        lines.append(
            "- MD&A narrative: based on summarized content from Item 7, Management’s Discussion and Analysis "
            "of Financial Condition and Results of Operations."
        )

    if "memo" in intents:
        lines.append(
            "- Overall assessment: synthesized from the extracted financial metrics, risk summary, and "
            "the generated investment memo derived from the Form 10-K."
        )

    if "stock" in intents:
        lines.append(
            "- Current stock price and intraday data: obtained from a live market data MCP tool "
            "(deterministic API call)."
        )

    # Section-based sources (only those actually used)
    if sections_index_filtered:
        for sid, meta in sections_index_filtered.items():
            label = meta.get("label") or meta.get("title") or "(unnamed section)"
            start_page = meta.get("start_page")
            if start_page is not None:
                lines.append(
                    f"- {label} (section id: {sid}, approx. page {start_page} of the Form 10-K)."
                )
            else:
                lines.append(
                    f"- {label} (section id: {sid} in the Form 10-K)."
                )

    if not lines:
        # Extreme fallback: no specific signals, but still indicate source
        lines.append(
            "- Structured evidence extracted from the parsed Form 10-K filing for the company."
        )

    return "\n".join(lines)


# ---------------- Prompt Builder (section-aware + intent-aware) -----------

def _build_multi_source_prompt(
    question: str,
    company: str,
    key_period: str,
    intents: Set[str],
    sections_index_filtered: Optional[Dict[str, Any]],
) -> str:
    """
    Build instructions for Gemini that can handle ANY combination of:
      - financials
      - risks
      - MD&A
      - memo / overall
      - stock (live market context)
    and that is explicitly aware of which 10-K sections are in scope.
    """

    # Human-friendly labels for headings
    intent_labels = []
    if "financials" in intents:
        intent_labels.append("Financial Highlights")
    if "risks" in intents:
        intent_labels.append("Main Risks")
    if "mdna" in intents:
        intent_labels.append("MD&A Insights")
    if "memo" in intents:
        intent_labels.append("Overall Assessment")
    if "stock" in intents:
        intent_labels.append("Stock Price / Live Market Context")

    sections_desc = ", ".join(intent_labels) if intent_labels else "Overall Answer"

    base = f"""
You are a careful financial analysis assistant answering questions
about a public company's 10-K filing and, when provided, live market data.

User question:
{question}

Company: {company}
Latest period: {key_period}

The user appears to be asking about: {sections_desc}.
""".strip()

    # Intent awareness text
    intents_list = sorted(list(intents)) if intents else ["memo"]
    intents_text = "Detected intents: " + ", ".join(intents_list)

    # Section awareness text
    if sections_index_filtered:
        sec_lines: List[str] = []
        for sid, meta in sections_index_filtered.items():
            label = meta.get("label") or meta.get("title") or "(no label)"
            approx_len = meta.get("approx_char_len", "?")
            sec_lines.append(f"- {sid}: {label} (approx {approx_len} chars)")
        sections_text = (
            "You are allowed to use ONLY the following 10-K sections:\n"
            + "\n".join(sec_lines)
        )
    else:
        sections_text = (
            "No section metadata was provided; answer using only the JSON evidence."
        )

    # Route-specific guidance
    route_extra = ""
    if "financials" in intents:
        route_extra += """
FINANCIAL GUIDANCE:
- Focus on metrics such as revenue, net income, margins, cash flow, and year-over-year changes.
- Use numbers ONLY from the evidence JSON.
- Describe trends briefly (e.g., whether growth is accelerating or slowing).
""".strip()

    if "risks" in intents:
        route_extra += """

RISK FACTORS GUIDANCE:
- Use top risks from the evidence JSON (Item 1A) and any summarized headwinds.
- Group related risks and explain briefly why they matter financially.
- Do NOT invent new risk categories or scenarios not present in the evidence.
""".strip()

    if "mdna" in intents:
        route_extra += """

MD&A GUIDANCE:
- Focus on management’s discussion of headwinds/tailwinds and narrative themes.
- Summarize how management explains recent performance and key drivers.
- Avoid re-quoting long passages; summarize in your own words.
""".strip()

    if "memo" in intents:
        route_extra += """

MEMO / OVERALL GUIDANCE:
- Provide a concise, balanced synthesis.
- Connect financial performance with risks and MD&A themes where appropriate.
- Do NOT give "buy/sell" recommendations or investment advice.
""".strip()

    if "stock" in intents:
        route_extra += """

STOCK PRICE / LIVE MARKET GUIDANCE:
- If "live_market_data" is present in the EVIDENCE JSON, report:
    * current_price
    * change and change_percent
    * open, high, low
    * as_of timestamp
  in a clean, well-formatted way (no broken characters or missing spaces).
- When the question combines STOCK with FINANCIALS:
    * Explain how revenue growth, profitability, margins, and cash flow 
      generally influence valuation and investor sentiment.
    * Connect positive performance (e.g., strong YoY net income growth) 
      to upward valuation pressures.
    * Connect weakening metrics (e.g., slower FCF growth) 
      to potential investor concerns or valuation compression.
    * Emphasize that fundamentals influence medium-/long-term price levels.
- When the question combines STOCK with RISKS:
    * Explain how key risks (macroeconomic, supply chain, competition, etc.) 
      typically affect investor confidence, volatility, or downside pressure.
- When STOCK is combined with either FINANCIALS or RISKS:
    * Interpret the CURRENT daily move (positive or negative), 
      but do NOT assert direct causality from a single day’s price action.
    * Clarify that short-term movements often reflect market sentiment, 
      broader macro conditions, or news unrelated to fundamentals.
- ALWAYS keep causality statements moderate:
    * Use language like "tends to", "can influence", "may support", 
      "often leads to", "could contribute", etc.
- NEVER imply certainty such as "this caused the stock to move" unless the 
  evidence JSON explicitly states this.
""".strip()

    if not route_extra:
        route_extra = """
GENERAL GUIDANCE:
- Provide a high-level explanation using whatever evidence is present.
- Do not guess or interpolate missing values.
""".strip()

    rules = """
You MUST follow these rules:
1. Use ONLY the data provided in the JSON evidence and the allowed 10-K sections.
2. Never invent numbers, dates, or risk factors.
3. If information is missing, say so explicitly and answer partially.
4. Structure your answer into clearly labeled sections for each relevant aspect, e.g.:
   - "Financial Highlights"
   - "Main Risks"
   - "MD&A Insights"
   - "Overall Assessment"
   - "Stock Price / Live Market Context"
   Only include sections that are actually relevant to the question.
5. Keep answers concise but clear (roughly 200–350 words).
6. Do NOT provide investment advice (no "buy/sell" recommendations).
7. Do NOT invent or guess specific product model names (e.g., “iPhone 16”, “Apple Watch Series 10”) unless they appear explicitly in the evidence JSON or snippet text.
8. If the evidence JSON includes a field called "memory_snippets", treat these as summaries or excerpts from past analyses or memos for this user. You may use them as additional context, especially when the question refers to prior runs or previous conclusions, but do not contradict newer structured metrics.
9. If the evidence JSON includes a field called "recent_qa", it contains up to the last two question/answer pairs from THIS conversation session. Use them only to:
    - Resolve pronouns or references like "that", "those risks", or "the previous answer".
    - Maintain narrative consistency with what you just explained.
    Do NOT override or contradict structured financial metrics or the latest memo using these; they are conversational context only.
10. If the evidence JSON includes a field called "live_market_data", it contains the CURRENT stock price and market capitalization from a deterministic tool.
    - You MAY quote these numbers (price and market cap) exactly as they appear in "live_market_data".
    - Do NOT change, round, or recalculate them into new numeric values.
    - You may describe them qualitatively (e.g., "multi-trillion market cap") but any specific numeric value must match the JSON exactly.
11. If the evidence JSON includes a field called "conversation_summary", treat it as a compact recap of earlier conversation with this user about this filing. You may use it to:
    - Understand references like "as you said earlier" or "those risks you mentioned".
    - Avoid repeating the same explanations at length.
    Do NOT treat it as more authoritative than structured financial metrics or the latest 'recent_qa' answers.
""".strip()

    final_prompt = f"""
{base}

{intents_text}

{sections_text}

{route_extra}

{rules}

Your job:
- Answer the user's question faithfully.
- Use bullet points or small paragraphs.
- NEVER use information outside the evidence JSON and the allowed sections.
""".strip()

    return final_prompt


# ---------------- LLM Call Wrapper ---------------------------------------

def _call_gemini_for_qa(
    prompt: str,
    question: str,
    evidence: Dict[str, Any],
) -> str:
    """
    Call Gemini to answer the question given the instructions and
    the evidence JSON.
    """
    evidence_json = _compact_json(evidence)

    resp = _GEMINI_CLIENT.models.generate_content(
        model=_GEMINI_MODEL_QA,
        contents=[
            prompt,
            "\n\n--- QUESTION (repeat for clarity) ---\n",
            question,
            "\n\n--- EVIDENCE (JSON) ---\n",
            evidence_json,
        ],
    )

    text = getattr(resp, "text", None)
    if not text:
        text = str(resp)

    return text.strip()


# ---------------------------------------------------------------------------
# QaAgent
# ---------------------------------------------------------------------------

class QaAgent(BaseAgent):
    """
    Q&A Agent (dynamic multi-aspect, section-aware version) with
    contextual engineering + numeric hallucination guard:

    - Keeps a rolling conversation_summary in state.
    - Keeps only the last 2 Q&A pairs in qa_history.
    - Uses memory_snippets from ADK memory_service.
    - Uses sections_index map + intents for routing.
    - Validates money-like numbers against structured metrics and
      appends a short hallucination footer.
    """

    async def _run_async_impl(self, ctx: InvocationContext):
        state: Dict[str, Any] = ctx.session.state

        # Rolling summary from previous turns (may be empty)
        existing_summary: str = state.get("conversation_summary") or ""

        # Compact history (last 2 Q&A) from previous turns
        qa_history_raw = state.get("qa_history") or []
        if not isinstance(qa_history_raw, list):
            qa_history_raw = []

        # Prefer a fresh question from the invocation context, but allow
        # falling back to a previous one in state for manual testing.
        question = getattr(ctx, "new_message", None) or state.get("qa_last_question") or ""
        question_str = str(question)
        # Optional minor cleanup
        question_str = " ".join(question_str.split())

        if DEBUG_QA_AGENT:
            print("\n[QaAgent] START")
            print("[QaAgent] state keys BEFORE:", list(state.keys()))
            print(f"[QaAgent] raw question: {question_str!r}")

        if not question_str.strip():
            if DEBUG_QA_AGENT:
                print("[QaAgent] No question provided; skipping answer generation.")
                print("[QaAgent] END\n")
            # We still emit an event (no-op) to keep the runner happy.
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={}),
            )
            return

        # --- Detect intents (hybrid) ---
        intents = _detect_intents(question_str)
        if DEBUG_QA_AGENT:
            print("[QaAgent] detected intents:", intents)

        company = state.get("company") or "Unknown company"
        key_period = state.get("key_period") or "latest year"

        # --- Live Market Data (from MCP live_quote) ---
        live_quote = state.get("live_quote") or {}
        wants_live = _wants_live_quote(question_str)
        if DEBUG_QA_AGENT:
            print("[QaAgent] wants_live_quote:", wants_live)
            print("[QaAgent] live_quote present?:", bool(live_quote))

        # Treat "stock" as a first-class intent when the question clearly
        # refers to the current stock price / live market.
        if wants_live:
            intents.add("stock")

        # Core 10-K intents: if any of these are present, it's NOT a pure
        # "just give me the quote" question.
        has_core_10k_intent = any(
            i in intents for i in ("financials", "risks", "mdna")
        )

        # 🔁 SHORT-CIRCUIT: direct live quote answer ONLY for pure
        # stock-price questions (no financials/risks/MD&A intent).
        pure_stock_question = wants_live and not has_core_10k_intent

        if pure_stock_question and live_quote:
            if DEBUG_QA_AGENT:
                print("[QaAgent] route: live_quote (short-circuit)")

            answer = _format_live_quote_answer(live_quote, company=company)

            sources_text = (
                "- Current stock price and intraday data: obtained from a live "
                "market data MCP tool (deterministic API call)."
            )

            full_answer = (
                answer
                + "\n\n---\n**Sources used by the agent (for traceability):**\n"
                + sources_text
            )

            # Build a new history entry for this turn
            new_history_entry = {
                "q": question_str,
                "a": full_answer,
                "route": "live_quote",
            }

            qa_history_for_summary = qa_history_raw + [new_history_entry]
            # Update rolling summary using existing + this latest Q&A
            new_summary = _update_conversation_summary(
                existing_summary,
                qa_history_for_summary,
            )

            # Compact history: keep only the last 2 entries
            qa_history_compact = qa_history_for_summary[-2:]

            state_delta: Dict[str, Any] = {
                "qa_last_question": question_str,
                "qa_last_route": "live_quote",
                "qa_last_evidence": {
                    "live_market_data": live_quote,
                    "wants_live_quote": wants_live,
                },
                "qa_last_answer": full_answer,
                "qa_last_sources_text": sources_text,
                "qa_history": qa_history_compact,
                "conversation_summary": new_summary,
            }

            if DEBUG_QA_AGENT:
                snippet = full_answer[:400].replace("\n", " ")
                print(
                    "[QaAgent] answer snippet:",
                    snippet + ("..." if len(full_answer) > 400 else ""),
                )
                print("[QaAgent] END\n")

            yield Event(
                author=self.name,
                actions=EventActions(state_delta=state_delta),
            )
            return

        # --- Section index (10-K map) ---
        sections_index_raw = state.get("sections_index") or {}
        sections_index_norm = _normalize_sections_index(sections_index_raw)
        sections_index_filtered = _filter_sections_index_by_intents(
            sections_index_raw, intents
        )

        if DEBUG_QA_AGENT:
            print("[QaAgent] sections_index size:", len(sections_index_norm))
            print("[QaAgent] sections_index_filtered size:", len(sections_index_filtered))

        # 🔁 Fallback: if nothing matched AND this isn't a pure financials question,
        # try best-effort lexical retrieval from sections_index.
        if (
            not sections_index_filtered
            and "financials" not in intents
            and sections_index_norm
        ):
            fallback = _fallback_sections_for_question(
                question_str,
                sections_index_norm,
                top_k=5,
                min_score=0.15,
            )
            if DEBUG_QA_AGENT:
                print("[QaAgent] fallback sections used:", len(fallback))
            sections_index_filtered = fallback

        # --- Financial metrics ---
        financial_metrics = state.get("financial_metrics") or {}
        per_period = financial_metrics.get("per_period") or {}
        metrics_for_period = per_period.get(state.get("key_period"), {})

        # --- Risks / MD&A ---
        risk_summary = state.get("risk_mdna_summary") or {}
        item_1a_text = state.get("item_1a_text") or ""
        mdna_text = state.get("mdna_text") or ""

        # --- Memo ---
        memo_text = state.get("investment_memo_markdown") or ""

        # --- Memory: retrieve relevant past snippets (if memory_service configured) ---
        memory_snippets: List[str] = []
        try:
            memory_service = getattr(ctx, "memory_service", None)
            if memory_service is not None:
                search_response = await memory_service.search_memory(
                    app_name=ctx.app_name,
                    user_id=ctx.user_id,
                    query=question_str,
                )
                if DEBUG_QA_AGENT:
                    print("[QaAgent] memory search hits:", len(search_response.memories))
                for mem in search_response.memories[:3]:
                    if mem.content and mem.content.parts:
                        txt = mem.content.parts[0].text or ""
                        if txt:
                            # Keep snippets bounded so evidence doesn't explode
                            memory_snippets.append(txt[:2000])
        except Exception as e:
            if DEBUG_QA_AGENT:
                print(f"[QaAgent] memory search error: {e}")

        # --- Conversation window: last-2 Q&A pairs from this session (BEFORE this answer) ---
        recent_qa: List[Dict[str, str]] = []
        if isinstance(qa_history_raw, list):
            for h in qa_history_raw[-2:]:
                if not isinstance(h, dict):
                    continue
                q = h.get("q")
                a = h.get("a")
                if isinstance(q, str) and isinstance(a, str):
                    recent_qa.append({"q": q, "a": a})

        if DEBUG_QA_AGENT:
            print(f"[QaAgent] recent_qa count (from history): {len(recent_qa)}")

        def _truncate(text: str, max_chars: int = 6000) -> str:
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "\n\n[TRUNCATED]\n"

        evidence: Dict[str, Any] = {
            "company": company,
            "key_period": key_period,
            "intents": sorted(list(intents)),
            "sections_index_filtered": sections_index_filtered,
            "sections_index_size": len(sections_index_norm),

            "financial_metrics_latest": metrics_for_period,
            "financial_metrics_all": financial_metrics,

            "risk_mdna_summary": risk_summary,
            "item_1a_text_snippet": _truncate(item_1a_text, 6000),
            "mdna_text_snippet": _truncate(mdna_text, 6000),

            "investment_memo_snippet": _truncate(memo_text, 6000),

            # Long-term memory snippets from previous runs (optional)
            "memory_snippets": memory_snippets,

            # Conversation-window context: last-2 Q&A pairs in this session
            "recent_qa": recent_qa,

            # Rolling summary up to BEFORE this answer
            "conversation_summary": existing_summary,

            # Live market data from MCP (if present)
            "live_market_data": live_quote,
            "wants_live_quote": wants_live,
        }

        # Build prompt that explains the multi-aspect, section-aware logic
        prompt = _build_multi_source_prompt(
            question_str,
            company,
            key_period,
            intents,
            sections_index_filtered,
        )

        # Call Gemini for main answer
        raw_answer = _call_gemini_for_qa(prompt, question_str, evidence)

        # ---------- ✅ HALLUCINATION GUARD HOOK ----------
        # Only attempt numeric validation if the answer actually has money-like numbers
        if _has_money_like_numbers(raw_answer):
            try:
                validation = validate_money_values(raw_answer, evidence)
                raw_answer = append_validation_summary(raw_answer, validation)
            except Exception as e:
                if DEBUG_QA_AGENT:
                    print(f"[QaAgent] numeric validation error (ignored): {e}")
        # -------------------------------------------------

        # Build citations / sources block
        sources_text = _build_citations_text(
            company=company,
            key_period=key_period,
            intents=intents,
            sections_index_filtered=sections_index_filtered,
        )

        full_answer = (
            raw_answer
            + "\n\n---\n**Sources used by the agent (for traceability):**\n"
            + sources_text
        )

        route_label = "auto_multi:" + ",".join(sorted(intents))

        if DEBUG_QA_AGENT:
            snippet = full_answer[:400].replace("\n", " ")
            print(f"[QaAgent] route: {route_label}")
            print(
                "[QaAgent] answer snippet:",
                snippet + ("..." if len(full_answer) > 400 else ""),
            )
            print("[QaAgent] END\n")

        # --- Update qa_history with this new Q&A pair (context compaction: keep last 2) ---
        new_history_entry = {
            "q": question_str,
            "a": full_answer,
            "route": route_label,
        }

        qa_history_raw.append(new_history_entry)

        # Update rolling summary (existing_summary + ALL Q&A we still keep in history)
        qa_history_for_summary = qa_history_raw.copy()
        new_summary = _update_conversation_summary(
            existing_summary,
            qa_history_for_summary,
        )

        # Keep only the last 2 entries in state
        qa_history_compact = qa_history_raw[-2:]

        state_delta: Dict[str, Any] = {
            "qa_last_question": question_str,
            "qa_last_route": route_label,
            "qa_last_evidence": evidence,
            "qa_last_answer": full_answer,
            "qa_last_sources_text": sources_text,
            "qa_history": qa_history_compact,
            "conversation_summary": new_summary,
        }

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )


qa_agent = QaAgent(
    name="qa_agent",
    description=(
        "Answers user questions using extracted financial metrics, "
        "Item 1A Risk Factors, MD&A headwinds/tailwinds, the memo, "
        "long-term memory snippets from prior analyses, a rolling "
        "conversation summary, a compact conversation window (last 2 "
        "Q&A pairs in this session), and optionally live market data "
        "from MCP (price + market cap). Supports any combination: "
        "financials, risks, MD&A, stock context, or all of them, and is explicitly "
        "section-aware using the 10-K map, with a fallback retrieval "
        "over sections_index for broad/narrative questions. Adds a "
        "numeric hallucination check for money values and appends a "
        "sources block for traceability to the underlying 10-K."
    ),
)
