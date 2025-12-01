# tools/adk_pdf_tools.py

"""
Deterministic HTML / iXBRL extractor for 10-K.

Public API:
    parse_10k_html_ixbrl(file_path: str) -> dict

Return shape (values are now FULLY SCALED to actual units, e.g. USD):

{
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "symbol": "AAPL",
    "cik": "0000320193",
    "facts": {
        "revenue": {
            "2022-09-24": 394328000000.0,   # USD
            "2021-09-25": 365817000000.0,
            "2020-09-26": 274515000000.0,
        },
        ...
    }
}

- Multi-period support using contextRef → period-end mapping.
- Uses XBRL `unitRef` + `decimals` to scale values (thousands / millions, etc.).
- No ADK, no LLM. Pure Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import re

from bs4 import BeautifulSoup  # beautifulsoup4

# Debug flag for numeric scaling
DEBUG_IXBRL_NUMERIC = True
DEBUG_IXBRL_NUMERIC_MAX_LOG = 10  # avoid spamming logs


# ============================
# Public API
# ============================

def parse_10k_html_ixbrl(file_path: str) -> Dict[str, Any]:
    """
    Main extraction function for 10-K HTML / iXBRL.

    - Reads HTML
    - Builds contextRef -> period_end_date map
    - Builds unitRef -> description map
    - Extracts company name
    - Extracts ticker / trading symbol
    - Extracts CIK
    - Extracts multi-period numeric facts (FULLY SCALED)
    """
    html = _read_file_text(file_path)
    soup = BeautifulSoup(html, "lxml")  # XML-ish but HTML parser works fine

    context_period_map = _build_context_period_map(soup)
    unit_map = _build_unit_map(soup)

    company_name = _extract_company_name(soup)
    ticker = _extract_trading_symbol(soup)
    cik = _extract_cik(soup)
    facts = _extract_key_facts_from_ixbrl(soup, context_period_map, unit_map)

    return {
        "company_name": company_name,
        "ticker": ticker,
        "symbol": ticker,  # convenience alias
        "cik": cik,
        "facts": facts,
    }


# ============================
# Helpers: File I/O
# ============================

def _read_file_text(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.read_text(encoding="utf-8", errors="ignore")


# ============================
# Helpers: Context → period map
# ============================

def _build_context_period_map(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Build mapping from contextRef (id) to period end date string: 'YYYY-MM-DD'.

    Looks for tags like:
        <xbrli:context id="FD2022"> ... <xbrli:period><xbrli:instant>2022-09-24</xbrli:instant>
    or:
        <xbrli:period><xbrli:endDate>...</xbrli:endDate>

    Returns:
        {
            "FD2022": "2022-09-24",
            "FD2021": "2021-09-25",
            ...
        }
    """
    context_period: Dict[str, str] = {}

    def is_context_tag(tag):
        return (
            tag.name
            and tag.has_attr("id")
            and tag.name.lower().endswith("context")
        )

    def is_instant_tag(tag):
        return tag.name and tag.name.lower().endswith("instant")

    def is_enddate_tag(tag):
        return tag.name and tag.name.lower().endswith("enddate")

    for ctx in soup.find_all(is_context_tag):
        ctx_id = ctx.get("id")
        if not ctx_id:
            continue

        # Try instant first (point-in-time period)
        instant = ctx.find(is_instant_tag)
        if instant and instant.string:
            context_period[ctx_id] = instant.string.strip()
            continue

        # Fallback: endDate
        end_date = ctx.find(is_enddate_tag)
        if end_date and end_date.string:
            context_period[ctx_id] = end_date.string.strip()

    return context_period


# ============================
# Helpers: Unit map (unitRef → description)
# ============================

def _build_unit_map(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Build a light mapping from unit id -> description text.

    Example:
        <xbrli:unit id="usd">
            <xbrli:measure>iso4217:USD</xbrli:measure>
        </xbrli:unit>

    or:
        <xbrli:unit id="usd_thousands">
            <xbrli:measure>iso4217:USD</xbrli:measure>
            <xbrli:measure>shares</xbrli:measure>
        </xbrli:unit>
    """
    unit_map: Dict[str, str] = {}

    def is_unit_tag(tag):
        return (
            tag.name
            and tag.has_attr("id")
            and tag.name.lower().endswith("unit")
        )

    for unit in soup.find_all(is_unit_tag):
        uid = unit.get("id")
        if not uid:
            continue
        desc = unit.get_text(" ", strip=True)
        unit_map[uid] = desc or ""

    if DEBUG_IXBRL_NUMERIC and unit_map:
        print("[parse_10k_html_ixbrl] Detected units:")
        for uid, desc in list(unit_map.items())[:10]:
            print(f"  - unit id={uid!r}, desc={desc!r}")

    return unit_map


# ============================
# Helpers: Company name
# ============================

def _extract_company_name(soup: BeautifulSoup) -> Optional[str]:
    """
    Try to extract a human-readable company name from the iXBRL/HTML.

    Priority:
      0) Any tag with name attr containing EntityRegistrantName (covers ix:nonNumeric / ix:nonFraction)
      1) Inline XBRL fact for EntityRegistrantName (dei/us-gaap)
      2) Plain XML tag for EntityRegistrantName
      3) <title> fallback, trimming 'Form 10-K' etc.
    """

    # 0) Generic tag with name attr containing EntityRegistrantName
    generic = soup.find(
        lambda tag: getattr(tag, "has_attr", lambda _ : False)("name")
        and "entityregistrantname" in str(tag.get("name", "")).lower()
    )
    if generic:
        txt = generic.get_text(strip=True)
        if txt:
            return txt

    # 1) Inline XBRL facts
    for fact in soup.find_all([
        "ix:nonFraction", "ix:nonfraction",
        "ix:nonNumeric", "ix:nonnumeric",
        "nonNumeric", "nonnumeric",
    ]):
        name_attr = (fact.get("name") or "").lower()
        if "entityregistrantname" in name_attr:
            txt = fact.get_text(strip=True)
            if txt:
                return txt

    # 2) Plain XML tags
    possible_tags = [
        "dei:EntityRegistrantName",
        "us-gaap:EntityRegistrantName",
        "entityregistrantname",
    ]
    for tag_name in possible_tags:
        tag = soup.find(tag_name)
        if tag:
            txt = tag.get_text(strip=True)
            if txt:
                return txt

    # 3) Fallback: document <title>, try to strip "Form 10-K" boilerplate
    title_tag = soup.find("title")
    if title_tag:
        title_text = title_tag.get_text(strip=True)
        if title_text:
            lower = title_text.lower()
            if "form 10-k" in lower:
                # e.g. "Apple Inc. - Form 10-K" -> "Apple Inc."
                idx = lower.find("form 10-k")
                raw_before = title_text[:idx]
                cleaned = raw_before.strip(" -–—")
                if cleaned:
                    return cleaned
            return title_text.strip()

    return None


# ============================
# Helpers: Trading symbol / ticker
# ============================

def _extract_trading_symbol(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract the primary trading symbol / ticker.

    Priority:
      0) Any tag with name attr containing TradingSymbol (inline ix)
      1) Plain XML tags for TradingSymbol
      2) Very light heuristic on visible text layout (optional)
    """

    # 0) Inline XBRL facts: ix:nonFraction / ix:nonNumeric name="dei:TradingSymbol"
    for fact in soup.find_all([
        "ix:nonFraction", "ix:nonfraction",
        "ix:nonNumeric", "ix:nonnumeric",
        "nonNumeric", "nonnumeric",
    ]):
        name_attr = (fact.get("name") or "").lower()
        if "tradingsymbol" in name_attr:
            txt = fact.get_text(strip=True)
            if txt:
                return txt

    # Generic: any tag with name attr containing TradingSymbol
    generic = soup.find(
        lambda tag: getattr(tag, "has_attr", lambda _ : False)("name")
        and "tradingsymbol" in str(tag.get("name", "")).lower()
    )
    if generic:
        txt = generic.get_text(strip=True)
        if txt:
            return txt

    # 1) Plain XML tags
    possible_tags = [
        "dei:TradingSymbol",
        "tradingsymbol",
    ]
    for tag_name in possible_tags:
        tag = soup.find(tag_name)
        if tag:
            txt = tag.get_text(strip=True)
            if txt:
                return txt

    # 2) Light heuristic: look for text "Trading Symbol" and grab next <td>/<span>
    label_candidates = soup.find_all(
        string=lambda s: isinstance(s, str) and "trading symbol" in s.lower()
    )
    for label in label_candidates:
        parent = getattr(label, "parent", None)
        if not parent:
            continue
        next_el = parent.find_next(["td", "span", "div"])
        if next_el:
            txt = next_el.get_text(strip=True)
            if txt and 1 <= len(txt) <= 10:
                return txt

    return None


# ============================
# Helpers: CIK
# ============================

def _extract_cik(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract SEC CIK (Central Index Key).

    Priority:
      0) Inline XBRL tags with name attr containing EntityCentralIndexKey
      1) Plain XML tags for EntityCentralIndexKey
      2) Very simple text heuristic on 'Central Index Key' label
    """

    # 0) Inline XBRL facts
    for fact in soup.find_all([
        "ix:nonFraction", "ix:nonfraction",
        "ix:nonNumeric", "ix:nonnumeric",
        "nonNumeric", "nonnumeric",
    ]):
        name_attr = (fact.get("name") or "").lower()
        if "entitycentralindexkey" in name_attr:
            txt = fact.get_text(strip=True)
            if txt:
                return txt

    # Generic: any tag with name attr containing EntityCentralIndexKey
    generic = soup.find(
        lambda tag: getattr(tag, "has_attr", lambda _ : False)("name")
        and "entitycentralindexkey" in str(tag.get("name", "")).lower()
    )
    if generic:
        txt = generic.get_text(strip=True)
        if txt:
            return txt

    # 1) Plain XML tags
    possible_tags = [
        "dei:EntityCentralIndexKey",
        "entitycentralindexkey",
    ]
    for tag_name in possible_tags:
        tag = soup.find(tag_name)
        if tag:
            txt = tag.get_text(strip=True)
            if txt:
                return txt

    # 2) Simple text heuristic: label "Central Index Key"
    label_candidates = soup.find_all(
        string=lambda s: isinstance(s, str) and "central index key" in s.lower()
    )
    for label in label_candidates:
        parent = getattr(label, "parent", None)
        if not parent:
            continue
        next_el = parent.find_next(["td", "span", "div"])
        if next_el:
            txt = next_el.get_text(strip=True)
            # CIK is usually numeric, up to 10 digits
            if txt and txt.replace(" ", "").replace("-", "").isdigit():
                return txt.strip()

    return None


# ============================
# Helpers: Numeric facts (multi-period, fully scaled)
# ============================
def _extract_key_facts_from_ixbrl(
    soup: BeautifulSoup,
    context_period_map: Dict[str, str],
    unit_map: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    Extract key us-gaap facts from ix:nonFraction tags.

    Values are scaled using:
      - XBRL 'decimals' attribute
      - unitRef description heuristics (thousand / million / billion)

    Returns a nested dict:
        {
            "revenue": {
                "2022-09-24": 394328000000.0,
                "2021-09-25": 365817000000.0,
            },
            "net_income": {
                "2022-09-24": 99803000000.0,
                ...
            },
            "operating_income": {
                "2022-09-24": ...,
                ...
            },
            "depreciation_amortization": {
                "2022-09-24": ...,
                ...
            },
            ...
        }
    """

    # Friendly key -> list of candidate XBRL concepts
    concept_map = {
        "revenue": [
            "us-gaap:Revenues",
            "us-gaap:SalesRevenueNet",
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        ],
        "total_current_assets": [
            "us-gaap:AssetsCurrent",
        ],
        "total_assets": [
            "us-gaap:Assets",
        ],
        "net_income": [
            "us-gaap:NetIncomeLoss",
            "us-gaap:ProfitLoss",
        ],
        "operating_cash_flow": [
            "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        ],
        "capital_expenditures": [
            "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
            "us-gaap:CapitalExpendituresIncurredButNotYetPaid",
        ],

        # --- NEW: operating income and D&A for true EBITDA ---
        "operating_income": [
            "us-gaap:OperatingIncomeLoss",
        ],
        # Prefer aggregate D&A concepts; many filers use one of these
        "depreciation_amortization": [
            "us-gaap:DepreciationDepletionAndAmortization",
            "us-gaap:DepreciationAndAmortization",
        ],

        # --- share-related concepts for valuation math ---

        "weighted_avg_shares_diluted": [
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
            "us-gaap:WeightedAverageNumberOfShareOutstandingDiluted",
        ],
        "weighted_avg_shares_basic": [
            "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
        ],
        "common_shares_outstanding": [
            "us-gaap:CommonStockSharesOutstanding",
            "us-gaap:EntityCommonStockSharesOutstanding",
        ],
        "shares_outstanding": [
            "us-gaap:CommonStockSharesOutstanding",
            "us-gaap:EntityCommonStockSharesOutstanding",
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
            "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
        ],
    }

    # Raw XBRL concept -> { period_end_date: scaled_value }
    raw_values_by_concept: Dict[str, Dict[str, float]] = {}

    ix_tags = soup.find_all(
        lambda tag: tag.name and tag.name.lower() in ("ix:nonfraction", "nonfraction")
    )

    log_count = 0

    for tag in ix_tags:
        name_attr = tag.get("name") or ""
        if not name_attr:
            continue

        context_ref = tag.get("contextref")
        if not context_ref:
            continue

        period_end = context_period_map.get(context_ref)
        if not period_end:
            # If we don't know the period, skip (keeps data clean)
            continue

        text_value = tag.get_text(strip=True)
        if not text_value:
            continue

        number = _safe_parse_number(text_value)
        if number is None:
            continue

        # --- Scaling from decimals ---
        decimals_attr = tag.get("decimals")
        decimals: Optional[int] = None
        if decimals_attr is not None:
            try:
                decimals = int(decimals_attr)
            except ValueError:
                decimals = None

        scale_decimals = 1.0
        if decimals is not None:
            if decimals < 0:
                # e.g. decimals = -6 → value is in millions (10^6)
                scale_decimals = 10.0 ** abs(decimals)
            elif decimals > 0:
                # decimals > 0 typically means more precision, but we treat
                # it as: actual_value = reported / (10^decimals)
                scale_decimals = 1.0 / (10.0 ** decimals)

        # --- Scaling from unitRef (thousand / million / billion hints) ---
        unit_ref = tag.get("unitref") or tag.get("unitRef") or ""
        unit_desc = (unit_map.get(unit_ref) or unit_ref or "").lower()

        scale_unit = 1.0
        if "thousand" in unit_desc:
            scale_unit = 1_000.0
        elif "million" in unit_desc:
            scale_unit = 1_000_000.0
        elif "billion" in unit_desc:
            scale_unit = 1_000_000_000.0

        scaled_value = number * scale_decimals * scale_unit

        # Store
        concept_dict = raw_values_by_concept.setdefault(name_attr, {})
        concept_dict[period_end] = scaled_value

        # Debug logging for a few facts
        if DEBUG_IXBRL_NUMERIC and log_count < DEBUG_IXBRL_NUMERIC_MAX_LOG:
            print(
                "[ixbrl] fact:",
                f"name={name_attr!r}, period={period_end}, "
                f"text={text_value!r}, raw={number}, "
                f"decimals={decimals}, unit_ref={unit_ref!r}, "
                f"unit_desc={unit_desc!r}, scaled={scaled_value}"
            )
            log_count += 1

    # Now map raw XBRL concepts into our friendly keys
    result: Dict[str, Dict[str, float]] = {}

    for friendly_key, concept_list in concept_map.items():
        merged: Dict[str, float] = {}
        for concept in concept_list:
            period_map = raw_values_by_concept.get(concept)
            if not period_map:
                continue
            # Merge periods (if overlapping, last concept in list wins)
            for period_end, value in period_map.items():
                merged[period_end] = value

        # Only include keys that have at least one period
        if merged:
            result[friendly_key] = merged

    if DEBUG_IXBRL_NUMERIC and result.get("revenue"):
        sample_periods = list(result["revenue"].keys())
        print("[ixbrl] Sample scaled revenue values:")
        for p in sample_periods[:3]:
            print(f"  period={p}, revenue={result['revenue'][p]}")

    return result

def _safe_parse_number(text: str) -> Optional[float]:
    """
    Convert strings like '394,328', '(10,000)', '$12.5' into a float.
    Returns None if it can't parse.

    NOTE: We deliberately do NOT interpret 'million'/'thousand' suffixes here.
    Scaling is handled by XBRL (decimals + unitRef) instead.
    """
    t = text.strip()

    # Handle parentheses as negative
    is_negative = t.startswith("(") and t.endswith(")")
    t = t.strip("()")

    # Remove currency symbols
    t = t.replace("$", "").replace("€", "").replace("£", "")

    # Remove commas
    t = t.replace(",", "").strip()

    # Simple float parse
    try:
        num = float(t)
        if is_negative:
            num = -num
        return num
    except ValueError:
        return None
