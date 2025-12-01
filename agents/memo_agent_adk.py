# agents/memo_agent_adk.py
from __future__ import annotations

import json
import re
from typing import AsyncGenerator, Dict, Any, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from google import genai  # ✅ use same style as RiskMdnaAgent

DEBUG_MEMO_AGENT = True

# Gemini client (picks up GOOGLE_API_KEY from env)
_GEMINI_CLIENT = genai.Client()
_GEMINI_MODEL_MEMO = "gemini-2.0-flash-lite"


# ---------------------------------------------------------------------------
# Helper: robustly extract FIRST JSON dict from a messy LLM string
# ---------------------------------------------------------------------------
def _extract_first_json_dict(text: str) -> Optional[Dict[str, Any]]:
    """
    Try very hard to extract the FIRST top-level JSON object from text.

    Strategy:
    1) Use json.JSONDecoder().raw_decode on the leading part of the string.
    2) If that fails, do a brace-depth scan starting from the first '{'
       and stop at the point where depth returns to zero.
    """
    if not isinstance(text, str):
        return None

    s = text.lstrip()
    if not s:
        return None

    # 1) Try using JSONDecoder.raw_decode (handles "JSON...extra stuff")
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Manual brace-depth scan from first '{'
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(s[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        return None

    return None


class MemoAgent(BaseAgent):
    """
    LLM-powered MemoAgent.

    Responsibilities:
    - Read deterministic financial metrics prepared by FinancialAgent
      from state["memo_financials"] (and/or state["financial_metrics"]).
    - Combine them with risk/MD&A text produced by RiskMdnaAgent.
    - Incorporate:
        * pre-computed company_overview from CompanyAgent
        * pre-computed market_analysis from MarketAnalysisAgent (TAM / trends / competitors)
        * live_stock quote from LiveMarketStateAgent
        * valuation_metrics from ValuationAgent (market cap, P/E, FCF yield)
    - Generate an explicit Investment Thesis (3–5 bullets) and store it.
    - Call Gemini to generate a structured investment memo.
    - Store the memo in state["investment_memo"] and thesis in state["investment_thesis"].
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        state: Dict[str, Any] = ctx.session.state

        if DEBUG_MEMO_AGENT:
            print("\n[MemoAgent] START")
            print("[MemoAgent] state keys BEFORE:", list(state.keys()))

        memo_financials = state.get("memo_financials") or {}
        if not memo_financials:
            if DEBUG_MEMO_AGENT:
                print(
                    "[MemoAgent] No 'memo_financials' found in state. "
                    "Ensure FinancialAgent ran before MemoAgent."
                )
                print("[MemoAgent] END\n")
            # Yield no-op so pipeline can continue
            yield Event(author=self.name, actions=EventActions())
            return

        # --- 1) Pull financial metrics and context from memo_financials / state ---

        financial_metrics: Dict[str, Any] = (
            memo_financials.get("financial_metrics")
            or state.get("financial_metrics")
            or {}
        )
        financials_raw: Dict[str, Any] = memo_financials.get("financials_raw") or {}
        key_period: Optional[str] = memo_financials.get("key_period") or state.get("key_period")

        company: Optional[str] = memo_financials.get("company")
        ticker: Optional[str] = memo_financials.get("ticker")
        symbol: Optional[str] = memo_financials.get("symbol")

        latest_metrics: Dict[str, Any] = memo_financials.get("latest_metrics") or {}
        yoy_metrics_flat: Dict[str, Any] = memo_financials.get("yoy_metrics") or {}

        per_period_old: Dict[str, Dict[str, float]] = {}
        yoy_struct_old: Dict[str, Dict[str, float]] = {}

        if isinstance(financial_metrics, dict):
            maybe_per_period = financial_metrics.get("per_period")
            maybe_yoy = financial_metrics.get("yoy")
            if isinstance(maybe_per_period, dict):
                per_period_old = maybe_per_period
            if isinstance(maybe_yoy, dict):
                yoy_struct_old = maybe_yoy

        # If latest_metrics still empty, derive from financial_metrics by key_period
        if not latest_metrics and key_period:
            # Old style: per_period dict
            if per_period_old:
                latest_metrics = per_period_old.get(key_period, {}) or {}
            else:
                # New style: financial_metrics is {period: {metrics}}
                if isinstance(financial_metrics, dict):
                    candidate = financial_metrics.get(key_period)
                    if isinstance(candidate, dict):
                        latest_metrics = candidate

        if DEBUG_MEMO_AGENT:
            print(f"[MemoAgent] key_period (from memo/state): {key_period}")
            print("[MemoAgent] latest_metrics keys:", list(latest_metrics.keys()))
            print("[MemoAgent] yoy_metrics_flat keys:", list(yoy_metrics_flat.keys()))
            print("[MemoAgent] yoy_struct_old keys:", list(yoy_struct_old.keys()))

        # --- 2) Pull risk / MD&A text from state (RiskAgent output) ---

        risk_mdna_summary = state.get("risk_mdna_summary")
        item_1a_text = state.get("item_1a_text")
        mdna_text = state.get("mdna_text")

        # --- 3) Pull company_overview, market_analysis, live_stock, valuation from state ---

        company_overview = state.get("company_overview") or ""
        live_stock = state.get("live_stock") or state.get("live_quote") or {}
        valuation = state.get("valuation_metrics") or {}

        # 🆕 market_analysis from MarketAnalysisAgent, robust parse
        raw_market_analysis = state.get("market_analysis")
        market_analysis: Dict[str, Any] = {}

        if DEBUG_MEMO_AGENT:
            print("[MemoAgent] raw_market_analysis type:", type(raw_market_analysis))
            if isinstance(raw_market_analysis, str):
                print("[MemoAgent] raw_market_analysis (first 200 chars):")
                print(raw_market_analysis[:200])

        if isinstance(raw_market_analysis, dict):
            market_analysis = raw_market_analysis
        elif isinstance(raw_market_analysis, str):
            parsed = _extract_first_json_dict(raw_market_analysis)
            if isinstance(parsed, dict):
                market_analysis = parsed
            else:
                if DEBUG_MEMO_AGENT:
                    print(
                        "[MemoAgent] market_analysis JSON could not be parsed; "
                        "using empty dict."
                    )
        else:
            if DEBUG_MEMO_AGENT:
                print(
                    "[MemoAgent] market_analysis not present or unexpected type; "
                    "using empty dict."
                )

        if DEBUG_MEMO_AGENT:
            print("[MemoAgent] market_analysis present? :", bool(market_analysis))
            print(
                "[MemoAgent] market_analysis keys   :",
                list(market_analysis.keys()) if isinstance(market_analysis, dict) else "N/A",
            )

        # --- 4) Build structured memo_input for prompt construction ---

        def pct(x: Optional[float]) -> Optional[str]:
            """Convert a fraction (0–1) to '12.3%' string, or keep None."""
            if x is None:
                return None
            return f"{x * 100:.1f}%"

        def pick_growth(metric_name: str) -> Optional[str]:
            """
            Helper to pick YoY growth for the key_period.

            New style: growth metrics are stored directly in latest_metrics,
            e.g., 'revenue_growth', 'net_income_growth', etc.

            Old style: yoy_struct_old is {metric_name: {period: value}}.
            """
            if not key_period:
                return None

            # New style: direct value on latest_metrics
            direct_val = latest_metrics.get(metric_name)
            if isinstance(direct_val, (int, float)):
                return pct(direct_val)

            # Old style: metric_name → {period: value}
            metric_growth = yoy_struct_old.get(metric_name, {})
            if isinstance(metric_growth, dict):
                val = metric_growth.get(key_period)
                if isinstance(val, (int, float)):
                    return pct(val)

            return None

        yoy_strings: Dict[str, Optional[str]] = {
            "revenue_growth_pct": pick_growth("revenue_growth"),
            "net_income_growth_pct": pick_growth("net_income_growth"),
            "fcf_growth_pct": pick_growth("fcf_growth"),
            "ebitda_growth_pct": pick_growth("ebitda_growth"),
        }

        memo_input: Dict[str, Any] = {
            "company": company,
            "ticker": ticker or symbol,
            "key_period": key_period,
            "metrics": {
                "revenue": latest_metrics.get("revenue"),
                "net_income": latest_metrics.get("net_income"),
                "gross_margin_pct": pct(latest_metrics.get("gross_margin")),
                "operating_margin_pct": pct(latest_metrics.get("operating_margin")),
                "ebitda_margin_pct": pct(latest_metrics.get("ebitda_margin")),
                "net_margin_pct": pct(latest_metrics.get("net_margin")),
                "fcf": latest_metrics.get("fcf"),
                "fcf_margin_pct": pct(latest_metrics.get("fcf_margin")),
            },
            "yoy": yoy_strings,
            "risk_mdna_summary": risk_mdna_summary,
            "item_1a_text": item_1a_text,
            "mdna_text": mdna_text,
            "company_overview": company_overview,
            "live_stock": live_stock,
            "valuation": valuation,
            "market_analysis": market_analysis,  # 🆕 parsed dict
            # Optional: raw facts for debugging / traceability
            "financials_raw": financials_raw,
        }

        if DEBUG_MEMO_AGENT:
            print("[MemoAgent] key_period:", key_period)
            print("[MemoAgent] latest_metrics keys (final):", list(memo_input["metrics"].keys()))
            print("[MemoAgent] YoY strings:", memo_input["yoy"])
            print("[MemoAgent] company_overview present?:", bool(company_overview))
            print("[MemoAgent] valuation present?:", bool(valuation))
            print("[MemoAgent] memo_input prepared.")

        # --- 5) First LLM call: explicit Investment Thesis (3–5 bullets) ---

        investment_thesis = _generate_investment_thesis(memo_input)
        memo_input["investment_thesis"] = investment_thesis

        if DEBUG_MEMO_AGENT:
            print("[MemoAgent] investment_thesis generated (first 400 chars):")
            print(investment_thesis[:400])

        # --- 6) Second LLM call: full memo that reuses thesis, overview, market & valuation ---

        prompt = _build_memo_prompt(memo_input)

        resp = _GEMINI_CLIENT.models.generate_content(
            model=_GEMINI_MODEL_MEMO,
            contents=[prompt],
        )

        text = resp.text if hasattr(resp, "text") else str(resp)
        memo_text = (text or "").strip()

        if not memo_text:
            memo_text = "MemoAgent was unable to generate a memo from the provided inputs."

        state_delta = {
            "investment_memo": memo_text,
            "investment_thesis": investment_thesis,
        }

        if DEBUG_MEMO_AGENT:
            print("[MemoAgent] Memo generated (first 400 chars):")
            print(memo_text[:400])
            print("[MemoAgent] END\n")

        # --- 7) Yield event to persist memo & thesis into session state ---

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )


def _generate_investment_thesis(memo_input: Dict[str, Any]) -> str:
    """
    Generate a 3–5 bullet Investment Thesis using ONLY:
    - financial metrics
    - YoY growth
    - risk_mdna_summary
    - item_1a_text
    - live_stock (if present)
    - deterministic valuation metrics (from valuation_metrics, if available)

    NOTE: We deliberately do NOT rely on TAM/SAM/SOM here; that is used in the main memo,
    and protected by MarketAnalysisAgent anti-hallucination rules.
    """
    company = memo_input.get("company") or "the company"
    ticker = memo_input.get("ticker") or ""
    key_period = memo_input.get("key_period") or "the latest fiscal year"

    m = memo_input.get("metrics") or {}
    yoy = memo_input.get("yoy") or {}

    risk_mdna_summary = memo_input.get("risk_mdna_summary")
    item_1a_text = memo_input.get("item_1a_text")
    live_stock = memo_input.get("live_stock") or {}
    valuation = memo_input.get("valuation") or {}

    current_price = live_stock.get("current_price")
    as_of = live_stock.get("as_of") or live_stock.get("timestamp")

    pe = valuation.get("pe")
    fcf_yield_pct = valuation.get("fcf_yield_pct")

    thesis_prompt = f"""
You are an equity research analyst.

Using ONLY the following inputs, generate a clear **Investment Thesis** for {company} ({ticker}):

Period analyzed: {key_period}

Core financial metrics:
- Revenue: {m.get("revenue")}
- Net income: {m.get("net_income")}
- Gross margin: {m.get("gross_margin_pct")}
- Operating margin: {m.get("operating_margin_pct")}
- EBITDA margin: {m.get("ebitda_margin_pct")}
- Net margin: {m.get("net_margin_pct")}
- Free cash flow (FCF): {m.get("fcf")}
- FCF margin: {m.get("fcf_margin_pct")}

Year-over-year growth:
- Revenue growth: {yoy.get("revenue_growth_pct")}
- Net income growth: {yoy.get("net_income_growth_pct")}
- FCF growth: {yoy.get("fcf_growth_pct")}
- EBITDA growth: {yoy.get("ebitda_growth_pct")}

Risk context:
- Risk/MD&A summary (pre-processed): {risk_mdna_summary}
- Item 1A raw text (if present): {item_1a_text}

Live stock quote (if available):
- Current share price: {current_price}
- Price timestamp: {as_of}

Deterministic valuation metrics (if available; do not change these values):
- P/E ratio: {pe}
- FCF yield: {fcf_yield_pct}

TASK:
Produce 3–5 bullet points that form an **Investment Thesis**.

Each bullet should:
- Be clearly **bullish** or **bearish** (you can label them like "[Bull]" or "[Bear]" at the start).
- Be grounded directly in the provided metrics, growth, risks, or valuation context.
- Capture decision-useful arguments such as: strength/weakness of moat, profitability quality,
  balance sheet / cash generation, growth runway, risk profile, and whether the valuation
  (P/E, FCF yield) seems supportive or stretched.

Rules:
- Do NOT invent any new specific numbers beyond those provided above.
- If some metrics or valuation values are missing, acknowledge uncertainty instead of guessing.
- Be balanced: it is fine if some bullets are bullish and some are bearish.
- Output ONLY the bullet points in markdown (no extra commentary before or after).
""".strip()

    try:
        resp = _GEMINI_CLIENT.models.generate_content(
            model=_GEMINI_MODEL_MEMO,
            contents=[thesis_prompt],
        )
        thesis_text = getattr(resp, "text", "") or ""
    except Exception as e:
        thesis_text = ""
        if DEBUG_MEMO_AGENT:
            print(f"[MemoAgent] Error generating investment_thesis: {e}")

    thesis_text = thesis_text.strip()
    if not thesis_text:
        thesis_text = (
            "- [Info] An explicit investment thesis could not be generated from the available "
            "metrics, risk text, and valuation inputs. Additional context may be required."
        )

    return thesis_text


def _build_memo_prompt(memo_input: Dict[str, Any]) -> str:
    """
    Turn structured metrics + risk text + company_overview + market_analysis +
    investment_thesis + deterministic valuation metrics into a single LLM prompt string.
    """
    company = memo_input.get("company") or "the company"
    ticker = memo_input.get("ticker") or ""
    key_period = memo_input.get("key_period") or "the latest fiscal year"

    m = memo_input.get("metrics") or {}
    yoy = memo_input.get("yoy") or {}

    risk_mdna_summary = memo_input.get("risk_mdna_summary")
    item_1a_text = memo_input.get("item_1a_text")
    mdna_text = memo_input.get("mdna_text")
    company_overview = memo_input.get("company_overview") or ""
    live_stock = memo_input.get("live_stock") or {}
    investment_thesis = memo_input.get("investment_thesis") or ""
    valuation = memo_input.get("valuation") or {}
    market_analysis = memo_input.get("market_analysis") or {}

    current_price = live_stock.get("current_price")
    as_of = live_stock.get("as_of") or live_stock.get("timestamp")

    market_cap = valuation.get("market_cap")
    pe = valuation.get("pe")
    fcf_yield_pct = valuation.get("fcf_yield_pct")

    # unpack structured fields from market_analysis
    ma_company = market_analysis.get("company")
    ma_industry = market_analysis.get("industry")
    ma_market_size = market_analysis.get("market_size") or {}
    ma_growth = market_analysis.get("growth") or {}
    ma_comp = market_analysis.get("competitive_landscape") or {}
    ma_trends = market_analysis.get("industry_trends") or {}

    # 🔄 Align keys between MarketAnalysisAgent (TAM/SAM/SOM/CAGR)
    #    and MemoAgent prompt (tam_usd/sam_usd/som_usd/cagr_percent)
    tam_raw = ma_market_size.get("TAM")
    if tam_raw is None:
        tam_raw = ma_market_size.get("tam_usd")

    sam_raw = ma_market_size.get("SAM")
    if sam_raw is None:
        sam_raw = ma_market_size.get("sam_usd")

    som_raw = ma_market_size.get("SOM")
    if som_raw is None:
        som_raw = ma_market_size.get("som_usd")

    tam_usd = tam_raw if tam_raw is not None else "unknown"
    sam_usd = sam_raw if sam_raw is not None else "unknown"
    som_usd = som_raw if som_raw is not None else "unknown"
    tam_notes = ma_market_size.get("notes")

    cagr_raw = ma_growth.get("CAGR")
    if cagr_raw is None:
        cagr_raw = ma_growth.get("cagr_percent")

    cagr_percent = cagr_raw if cagr_raw is not None else "unknown"
    cagr_period = ma_growth.get("period", "unknown")

    return f"""
You are an equity research analyst writing a concise but insightful **public equity investment memo**
for a finance-literate audience.

Company: {company} ({ticker})
Analysis period: {key_period}

Core financial metrics for this period:
- Revenue: {m.get("revenue")}
- Net income: {m.get("net_income")}
- Gross margin: {m.get("gross_margin_pct")}
- Operating margin: {m.get("operating_margin_pct")}
- EBITDA margin: {m.get("ebitda_margin_pct")}
- Net margin: {m.get("net_margin_pct")}
- Free cash flow (FCF): {m.get("fcf")}
- FCF margin: {m.get("fcf_margin_pct")}

Year-over-year growth for the same period (if provided):
- Revenue growth: {yoy.get("revenue_growth_pct")}
- Net income growth: {yoy.get("net_income_growth_pct")}
- FCF growth: {yoy.get("fcf_growth_pct")}
- EBITDA growth: {yoy.get("ebitda_growth_pct")}

Live stock context (if available):
- Current share price: {current_price}
- Price timestamp: {as_of}

Deterministic valuation metrics (do NOT change these values):
- Market capitalization: {market_cap}
- P/E ratio: {pe}
- FCF yield: {fcf_yield_pct}

Risk and qualitative context:
- Risk/MD&A summary (pre-processed): {risk_mdna_summary}
- Item 1A raw text (if present): {item_1a_text}
- MD&A raw text (if present): {mdna_text}

Pre-computed Company Overview section (from a separate agent):
{company_overview}

Pre-computed Market Analysis (from MarketAnalysisAgent; do NOT contradict these facts):
- Market analysis company focus: {ma_company}
- Industry: {ma_industry}
- TAM (USD): {tam_usd}
- SAM (USD): {sam_usd}
- SOM (USD): {som_usd}
- TAM/SAM/SOM notes: {tam_notes}
- Market CAGR (%): {cagr_percent}
- CAGR period: {cagr_period}
- Competitive landscape object: {ma_comp}
- Industry trends object: {ma_trends}

Pre-computed Investment Thesis bullets (from a separate reasoning step):
{investment_thesis}

IMPORTANT ANTI-HALLUCINATION RULES FOR MARKET SECTION:
- Do NOT invent any new TAM, SAM, SOM, or market growth numbers.
- You may ONLY use the TAM/SAM/SOM and CAGR values provided above.
- If any of TAM/SAM/SOM are 'unknown', explicitly state in the memo that detailed market
  size is not available and do NOT fabricate or approximate numbers.
- You may qualitatively discuss growth, competition, and positioning based on the
  MarketAnalysisAgent summary and the 10-K text, but keep numbers consistent.
  STRICT RULE — DO NOT ADD HEADERS:
You MUST NOT include any analyst signature, model name, or label such as 
"Analyst:", "Bard:", "Gemini:", "GPT:", "AI:", or anything similar. 
Do NOT sign the memo. 
Do NOT add any attribution line.

TASK:
Write a professional, clearly structured **investment memo** with the following sections,
in this exact order and using markdown headings:

1. Executive Summary
   - 5–8 bullet points.
   - Clearly state what the company does, the period analyzed, 3–5 key financial highlights,
     2–3 key risks, and a one-sentence investment stance.

2. Company Overview
   - Start from the pre-computed 'Company Overview' section above.
   - You may lightly edit for clarity and flow, but do NOT contradict the provided overview.
   - Cover business model, products & services, revenue segments/geographies (if known),
     target customers, and competitive advantages.

3. Market Analysis
   - Use the pre-computed MarketAnalysisAgent output above AND the 10-K / MD&A context.
   - Discuss industry, major trends, competitive landscape, and how the company is positioned.
   - If TAM/SAM/SOM or detailed market size is 'unknown' above, explicitly say that
     the market size is not precisely quantified instead of inventing numbers.
   - If TAM/SAM/SOM and CAGR are provided, you may interpret them qualitatively
     (e.g., "large and growing market") but must NOT change the numeric values.

4. Financial Performance
   - Analyze revenue, margins, and YoY growth.
   - Comment on trends in profitability (gross, operating, EBITDA, net margins).
   - Highlight any notable strengths or weaknesses in the financial profile.

5. Cash Flow and Capital Allocation
   - Analyze FCF level and FCF margin.
   - Comment on the quality and sustainability of cash generation.
   - If not explicitly known, you can qualitatively discuss likely capital allocation
     (debt reduction, buybacks, dividends, reinvestment), but label speculation clearly.

6. Risks & Mitigations
   - Use the risk/MD&A summary and raw text to identify major risk categories:
     operational, financial, regulatory, competitive, macro, etc.
   - For each major risk, briefly suggest potential mitigations or why it may be manageable,
     if that can be reasonably inferred from 10-K context.
   - Do NOT invent detailed mitigation programs that are not mentioned.

7. Investment Thesis
   - Start from the pre-computed Investment Thesis bullets above.
   - You may refine wording for clarity, remove duplicates, or slightly re-balance,
     but keep the core logic and do NOT contradict them.
   - Present 3–5 final bullets that summarize the strongest reasons to invest or not invest.

8. Valuation & Deal Terms
   - Use the deterministic valuation metrics (market cap, P/E, FCF yield) to comment on whether
     the company appears expensive, fairly valued, or cheap in a qualitative sense.
   - Do NOT change the numeric values of P/E or FCF yield; you may only interpret them.
   - If any of the values are missing, explicitly note that valuation analysis is limited
     and explain what would be needed for a fuller view (e.g., peer multiples, history).

9. Recommendation
   - Provide a clear labeled recommendation in the first line of this section, choosing exactly one of:
       - "Recommendation: BUY"
       - "Recommendation: HOLD"
       - "Recommendation: SELL"
       - or (if preferred) "Recommendation: INVEST", "DO NOT INVEST", or "INVEST WITH CONDITIONS".
   - Then justify this recommendation in 3–5 sentences, tying back to:
       - financial performance
       - risks
       - qualitative positioning
       - and the deterministic valuation metrics above (if available).

RULES:
- Do NOT invent specific numeric values (e.g., revenue, EPS, margins, TAM, P/E, FCF yield)
  that are not provided in the inputs or the pre-computed market analysis.
- If a metric or growth rate is missing, explicitly acknowledge that limitation and base your reasoning
  only on what is available.
- Prefer conservative, clearly caveated language over aggressive speculation.
- Keep the memo focused, analytical, and readable to a finance-literate audience.
- Preserve the overall structure and section order exactly as specified.
"""


memo_agent = MemoAgent(
    name="memo_agent",
    description=(
        "LLM-based memo writer that consumes deterministic financial metrics, "
        "pre-computed company overview, pre-computed market analysis (TAM / trends / competitors), "
        "live stock quote, and valuation metrics "
        "to produce a structured investment memo with an explicit investment thesis "
        "and recommendation."
    ),
)
