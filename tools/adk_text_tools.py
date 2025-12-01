from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, List, Any

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Basic file + heading helpers
# ---------------------------------------------------------------------------

def _read_file_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8", errors="ignore")


def _find_item_headings(full_text: str) -> List[Dict[str, Any]]:
    """
    Find 'Item X.' style headings in the flattened 10-K text.

    Returns a list of dicts:
      {
        "start": int,      # character index of heading
        "end": int,        # end of heading line
        "section_number": str,  # e.g. "1", "1A", "7", "8"
        "heading": str,    # full heading line text
        "section_id": str, # normalized id like "item_1a"
      }
    """

    # Allow patterns like:
    # "Item 1A. Risk Factors", "ITEM 7. Management's Discussion...", etc.
    pattern = re.compile(
        r"(item\s+(\d+[a]?)\.\s*[^\n]{0,120})",
        re.IGNORECASE,
    )

    matches: List[Dict[str, Any]] = []
    for m in pattern.finditer(full_text):
        heading_text = m.group(1).strip()
        number = m.group(2).strip()  # e.g., "1A", "7"
        section_id = f"item_{number.lower()}"  # "item_1a", "item_7"
        matches.append(
            {
                "start": m.start(1),
                "end": m.end(1),
                "section_number": number,
                "heading": heading_text,
                "section_id": section_id,
            }
        )

    # Sort by appearance in the document
    matches.sort(key=lambda x: x["start"])
    return matches


# ---------------------------------------------------------------------------
# 10-K section map builder (text + sections_index)
# ---------------------------------------------------------------------------

def build_10k_section_map(file_path: str) -> Dict[str, Any]:
    """
    Build a 'map' of the 10-K:

      - Parses HTML, flattens to text.
      - Finds all "Item X." headings.
      - Splits full text into sections per heading.

    Returns:
      {
        "full_text": "<entire flattened text>",
        "sections_by_id": {
          "item_1a": {"id": "item_1a", "label": "Item 1A. Risk Factors", "text": "..."},
          "item_7":  {...},
          "item_8":  {...},
          ...
        },
        "chunks": [
          {
            "id": "item_1a",
            "label": "Item 1A. Risk Factors",
            "section_number": "1A",
            "text": "...."
          },
          ...
        ],
        "sections_index": {
          "item_1a": {
            "id": "item_1a",
            "label": "Item 1A – Risk Factors",
            "section_type": "risks",
            "part": "PART I",
            "page_estimate": 9,
            "approx_char_len": 70825,
            "present_in_document": true,
          },
          ...
        }
      }
    """

    html = _read_file_text(file_path)
    soup = BeautifulSoup(html, "lxml")

    # Flatten to text with some newlines preserved
    full_text = soup.get_text("\n", strip=True)

    headings = _find_item_headings(full_text)
    chunks: List[Dict[str, Any]] = []

    if not headings:
        # If we can't detect any headings, return just full_text
        return {
            "full_text": full_text,
            "sections_by_id": {},
            "chunks": [],
            "sections_index": {},
        }

    # Build sections between headings
    for i, h in enumerate(headings):
        start = h["start"]
        if i + 1 < len(headings):
            end = headings[i + 1]["start"]
        else:
            end = len(full_text)

        section_text = full_text[start:end].strip()
        chunk = {
            "id": h["section_id"],
            "label": h["heading"],
            "section_number": h["section_number"],
            "text": section_text,
        }
        chunks.append(chunk)

    # Index by section_id
    sections_by_id: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        # If the same section_id appears multiple times, keep the longest text
        sid = c["id"]
        if sid not in sections_by_id or len(c["text"]) > len(sections_by_id[sid]["text"]):
            sections_by_id[sid] = c

    # Build a richer metadata index for section-aware routing.
    sections_index = _build_sections_index(sections_by_id)

    return {
        "full_text": full_text,
        "sections_by_id": sections_by_id,
        "chunks": chunks,
        "sections_index": sections_index,
    }


# ---------------------------------------------------------------------------
# 10-K sections_index with section_type
# ---------------------------------------------------------------------------

def _build_sections_index(
    sections_by_id: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build a structured metadata index for the 10-K, combining:

      - Static knowledge about standard 10-K items (labels, part, page).
      - Dynamic info from this specific document (approx_char_len, presence).

    This is what the Q&A layer and RiskMdnaAgent can use as the "10-K map".

    Key field for routing:
      section_type:
        - "risks"       -> Risk Factors (Item 1A)
        - "mdna"        -> MD&A (Item 7)
        - "financials"  -> Financial Statements (Item 8)
        - plus more granular types (business, legal, etc.) for future use.
    """

    def _entry(
        sec_id: str,
        default_label: str,
        section_type: str,
        part: str,
        page: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Use detected heading if present; otherwise fall back to default.
        detected = sections_by_id.get(sec_id) or {}
        detected_label = detected.get("label")
        text = detected.get("text") or ""
        return {
            "id": sec_id,
            "label": detected_label or default_label,
            "section_type": section_type,      # our own category (risks, mdna, financials, etc.)
            "part": part,                      # "PART I" / "PART II" / ...
            "page_estimate": page,             # from the index; just a hint
            "approx_char_len": len(text) if text else None,
            "present_in_document": bool(detected),
        }

    # Static 10-K index based on the Apple 10-K table of contents you pasted
    index: Dict[str, Any] = {
        # PART I
        "item_1": _entry(
            "item_1",
            "Item 1. Business",
            section_type="business",
            part="PART I",
            page=3,
        ),
        "item_1a": _entry(
            "item_1a",
            "Item 1A. Risk Factors",
            section_type="risks",
            part="PART I",
            page=9,
        ),
        "item_1b": _entry(
            "item_1b",
            "Item 1B. Unresolved Staff Comments",
            section_type="staff_comments",
            part="PART I",
            page=28,
        ),
        "item_1c": _entry(
            "item_1c",
            "Item 1C. Cybersecurity",
            section_type="cybersecurity",
            part="PART I",
            page=28,
        ),
        "item_2": _entry(
            "item_2",
            "Item 2. Properties",
            section_type="properties",
            part="PART I",
            page=29,
        ),
        "item_3": _entry(
            "item_3",
            "Item 3. Legal Proceedings",
            section_type="legal",
            part="PART I",
            page=51,
        ),
        "item_4": _entry(
            "item_4",
            "Item 4. Mine Safety Disclosures",
            section_type="safety",
            part="PART I",
            page=51,
        ),

        # PART II
        "item_5": _entry(
            "item_5",
            "Item 5. Market for the Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
            section_type="equity_market",
            part="PART II",
            page=53,
        ),
        "item_6": _entry(
            "item_6",
            "Item 6. [Reserved]",
            section_type="reserved",
            part="PART II",
            page=54,
        ),
        "item_7": _entry(
            "item_7",
            "Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations",
            section_type="mdna",
            part="PART II",
            page=54,
        ),
        "item_7a": _entry(
            "item_7a",
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
            section_type="market_risk",
            part="PART II",
            page=82,
        ),
        "item_8": _entry(
            "item_8",
            "Item 8. Financial Statements and Supplementary Data",
            section_type="financials",
            part="PART II",
            page=84,
        ),
        "item_9": _entry(
            "item_9",
            "Item 9. Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
            section_type="accounting_changes",
            part="PART II",
            page=137,
        ),
        "item_9a": _entry(
            "item_9a",
            "Item 9A. Controls and Procedures",
            section_type="controls",
            part="PART II",
            page=137,
        ),
        "item_9b": _entry(
            "item_9b",
            "Item 9B. Other Information",
            section_type="other_info",
            part="PART II",
            page=137,
        ),
        "item_9c": _entry(
            "item_9c",
            "Item 9C. Disclosure Regarding Foreign Jurisdictions That Prevent Inspections",
            section_type="foreign_disclosure",
            part="PART II",
            page=137,
        ),

        # PART III
        "item_10": _entry(
            "item_10",
            "Item 10. Directors, Executive Officers and Corporate Governance",
            section_type="directors_governance",
            part="PART III",
            page=137,
        ),
        "item_11": _entry(
            "item_11",
            "Item 11. Executive Compensation",
            section_type="compensation",
            part="PART III",
            page=138,
        ),
        "item_12": _entry(
            "item_12",
            "Item 12. Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
            section_type="ownership",
            part="PART III",
            page=138,
        ),
        "item_13": _entry(
            "item_13",
            "Item 13. Certain Relationships and Related Transactions, and Director Independence",
            section_type="relationships",
            part="PART III",
            page=138,
        ),
        "item_14": _entry(
            "item_14",
            "Item 14. Principal Accountant Fees and Services",
            section_type="accountant_fees",
            part="PART III",
            page=138,
        ),

        # PART IV
        "item_15": _entry(
            "item_15",
            "Item 15. Exhibits and Financial Statement Schedules",
            section_type="exhibits",
            part="PART IV",
            page=138,
        ),
        "item_16": _entry(
            "item_16",
            "Item 16. Form 10-K Summary",
            section_type="summary",
            part="PART IV",
            page=144,
        ),
    }

    # 🔹 OPTIONAL: add any extra detected sections that weren't in the static map
    for sid, sec in sections_by_id.items():
        if sid not in index:
            text = sec.get("text") or ""
            index[sid] = {
                "id": sid,
                "label": sec.get("label", sid),
                "section_type": "other",
                "part": "UNKNOWN",
                "page_estimate": None,
                "approx_char_len": len(text) if text else None,
                "present_in_document": True,
            }

    return index


# ---------------------------------------------------------------------------
# Extraction helper used by RiskMdnaAgent
# ---------------------------------------------------------------------------

def extract_10k_text_sections(file_path: str) -> Dict[str, Any]:
    """
    Compatibility helper used by RiskMdnaAgent.

    Returns the two key sections as raw text:
      - Item 1A. (Risk Factors)
      - Item 7.  (MD&A)

    PLUS a structured sections_index describing the 10-K sections
    (used later by Q&A for section-aware routing).
    """
    section_map = build_10k_section_map(file_path)
    sections_by_id = section_map.get("sections_by_id", {})
    sections_index = section_map.get("sections_index", {})

    item_1a_text = (sections_by_id.get("item_1a") or {}).get("text", "")  # Item 1A
    item_7_text = (sections_by_id.get("item_7") or {}).get("text", "")    # Item 7

    return {
        "item_1a_raw": item_1a_text,
        "item_7_raw": item_7_text,
        "sections_index": sections_index,
    }
