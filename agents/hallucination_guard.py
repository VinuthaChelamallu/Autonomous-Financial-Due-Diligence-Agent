"""
Hallucination Guardrail Utilities (Trust Layer)
----------------------------------------------
Extracts money-like numbers from answers and uses an LLM-as-judge
to qualitatively validate numeric statements against provided evidence.

Exposed functions:

- extract_money_values(answer) -> List[(value, unit)]
- validate_money_values(answer, evidence) -> Dict[str, Any]
- append_validation_summary(answer, validation) -> str
"""

from __future__ import annotations

import json
import re
from math import isfinite
from typing import Any, Dict, List, Tuple

from google import genai  # ✅ LLM client for "LLM-as-judge"

# Regex: captures $391.0 billion, 391 billion, $118 million, 118m, 12.5%
_MONEY_PATTERN = re.compile(
    r"(\$?\s*(\d+(?:\.\d+)?)\s*(billion|bn|million|m|thousand|k|%)?)",
    re.IGNORECASE,
)

# ✅ Gemini client (uses GOOGLE_API_KEY from env)
_GEMINI_CLIENT = genai.Client()

# Use the same model family you use elsewhere
_GEMINI_MODEL_VALIDATOR = "gemini-2.5-flash-lite"


def extract_money_values(answer: str) -> List[Tuple[float, str]]:
    """
    Extract money-like values from model answer.

    Returns:
        [(value: float, unit: str)]
    Where:
        - value is normalized to plain float (still in the unit's scale)
        - unit is one of ["billion", "bn", "million", "m", "thousand", "k", "%", ""]
    """
    results: List[Tuple[float, str]] = []

    if not answer:
        return results

    for match in _MONEY_PATTERN.finditer(answer):
        full, num_str, unit = match.groups()

        # Must contain "$", a percentage, or a unit word
        if "$" not in full and not unit:
            continue

        try:
            val = float(num_str)
        except ValueError:
            continue

        unit = (unit or "").lower()

        if not isfinite(val):
            continue

        results.append((val, unit))

    return results


def _summarize_evidence_for_llm(evidence: Dict[str, Any]) -> str:
    """
    Build a compact JSON/string summary of evidence for the LLM judge.

    Designed to work well with the compact validator_evidence from QaAgent:
        {
            "company": ...,
            "ticker": ...,
            "symbol": ...,
            "key_period": ...,
            "financial_metrics": {...},
            "valuation_metrics": {...},
            "live_stock": {...},
            "live_quote": {...},
        }

    If evidence is larger or differently structured, we still try to extract
    the most relevant numeric parts; otherwise we fall back to compact JSON.
    """
    if not evidence:
        return "null"

    # Prefer the numeric-heavy pieces if present
    keys_of_interest = [
        "company",
        "ticker",
        "symbol",
        "key_period",
        "financial_metrics",      # full metrics (all periods)
        "financial_metrics_all",  # legacy key if ever used
        "valuation_metrics",
        "live_stock",
        "live_quote",
        "live_market_data",       # in case it's present under this name
    ]

    slim: Dict[str, Any] = {}
    for k in keys_of_interest:
        if k in evidence:
            slim[k] = evidence[k]

    obj = slim or evidence

    try:
        # Bound the size so prompts don't explode
        return json.dumps(obj, default=str)[:8000]
    except TypeError:
        # Fallback: string representation
        return str(obj)[:8000]


def validate_money_values(answer: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM-as-judge numeric validation.

    Behavior:
    - Detects whether the answer contains any money-like / percentage values.
    - If none: returns has_numeric = False, ok = True (nothing to validate).
    - If yes:
        * Calls Gemini with:
            - the full answer text
            - a compact JSON summary of evidence
          and asks it to judge whether numeric statements are broadly consistent.
        * Expects JSON from the LLM of the form:
            {
              "ok": bool,
              "issues": ["short description", ...]
            }
        * We DO NOT do any deterministic comparison here; LLM is the judge.

    Returns a dict like:
        {
            "has_numeric": bool,
            "ok": bool,
            "issues": List[str],
            "judge_raw": Optional[str],  # raw LLM JSON/response for debugging
        }
    """
    money_values = extract_money_values(answer)
    has_numeric = bool(money_values)

    if not has_numeric:
        # No numeric claims detected → nothing to validate
        return {
            "has_numeric": False,
            "ok": True,
            "issues": [],
            "judge_raw": None,
        }

    evidence_summary = _summarize_evidence_for_llm(evidence or {})

    # LLM prompt: ask Gemini to be a strict, conservative numeric judge
    prompt = f"""
You are acting as a strict **numeric consistency judge** for a financial QA system.

You will be given:
1) A model-generated ANSWER that may contain numeric / money-like statements.
2) An EVIDENCE object (JSON) that contains structured financial metrics and
   optionally valuation data and live stock data for the same company.

TASK:
- Evaluate whether the numeric statements in ANSWER (money amounts, percentages,
  growth rates, valuation metrics, live price/market cap, etc.) are broadly
  consistent with EVIDENCE.
- Focus on *major* inconsistencies rather than tiny rounding differences.
- If you cannot tell from EVIDENCE whether a statement is correct, do NOT count
  that as an error; you may ignore it.

YOU MUST:
- Return ONLY a single JSON object with this structure (no extra text):
  {{
    "ok": bool,
    "issues": [
      "short human-readable description of a potential numeric issue",
      ...
    ]
  }}

Guidelines:
- "ok" should be false if you find at least one *material* inconsistency
  (e.g., revenue level or growth direction clearly contradicts EVIDENCE).
- "issues" should list at most 5 of the most important numeric issues.
- If everything appears broadly consistent OR there is not enough information
  to judge, set "ok": true and "issues": [].

ANSWER:
{answer}

EVIDENCE (JSON or compact representation):
{evidence_summary}
""".strip()

    try:
        resp = _GEMINI_CLIENT.models.generate_content(
            model=_GEMINI_MODEL_VALIDATOR,
            contents=[prompt],
        )
        raw_text = getattr(resp, "text", "") or ""
    except Exception as e:
        # Fail open but record the error as an "issue"
        return {
            "has_numeric": True,
            "ok": True,  # we don't block just because the judge failed
            "issues": [f"LLM validator error: {e}"],
            "judge_raw": None,
        }

    raw_text = raw_text.strip()

    # Try to parse JSON; if the model added extra text, attempt to extract the first {...} block
    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(raw_text)
    except Exception:
        try:
            match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
        except Exception:
            parsed = {}

    ok = parsed.get("ok", True) if isinstance(parsed, dict) else True
    issues = parsed.get("issues", []) if isinstance(parsed, dict) else []

    # Normalize issues to a list of strings
    if not isinstance(issues, list):
        issues = [str(issues)]
    issues = [str(x) for x in issues][:5]

    return {
        "has_numeric": True,
        "ok": bool(ok),
        "issues": issues,
        "judge_raw": raw_text or None,
    }


def append_validation_summary(answer: str, validation: Dict[str, Any]) -> str:
    """
    Append a short numeric-validation footer to the answer *only when*
    there were numeric claims to check.

    - If validation["has_numeric"] is False or missing → return `answer` unchanged.
    - If ok and no issues → append a green check line.
    - If not ok or issues present → append a warning block.
    """
    if not validation:
        return answer

    has_numeric = bool(validation.get("has_numeric"))
    if not has_numeric:
        # ✅ No numeric content → do NOT add any footer
        return answer

    ok = bool(validation.get("ok", True))
    issues: List[str] = validation.get("issues") or []

    lines: List[str] = []
    lines.append("\n\n**Numeric validation (LLM-as-judge check):**")

    if ok and not issues:
        lines.append(
            "✅ LLM-as-judge: numeric statements are broadly consistent with the "
            "available structured financial and market evidence (within reasonable "
            "rounding and scaling)."
        )
    else:
        lines.append(
            "⚠️ LLM-as-judge: some numeric statements may not match the underlying "
            "financial or live-market evidence. Please double-check critical figures."
        )
        for issue in issues[:5]:
            lines.append(f"- {issue}")

    return answer + "\n" + "\n".join(lines)
