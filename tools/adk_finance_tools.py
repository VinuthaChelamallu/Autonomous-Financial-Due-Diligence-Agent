# tools/adk_finance_tools.py

from __future__ import annotations

"""
ADK finance tools.

This module sits between:
  - ExtractionAgent (which parses the 10-K / iXBRL), and
  - financial_calcs.compute_core_metrics (which does deterministic math).

Responsibilities:
  1) Take the extraction payload (state["extraction"]).
  2) Build a canonical "facts" dictionary:
        {
            "revenue": {period -> value},
            "cost_of_sales": {period -> value},
            "operating_income": {period -> value},
            "net_income": {period -> value},
            "operating_cash_flow": {period -> value},
            "capital_expenditures": {period -> value},
            "interest_expense": {period -> value},
            "income_tax_expense": {period -> value},
            "depreciation_and_amortization": {period -> value},
        }
  3) Call compute_core_metrics(facts) and return the result.
"""

from typing import Any, Dict, Optional

from tools.financial_calcs import compute_core_metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_financial_metrics_from_extraction(extraction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper used by FinancialAgent:
      - Build canonical 'facts' from ExtractionAgent output.
      - Run deterministic core computations.

    Args:
        extraction: state["extraction"] from ExtractionAgent.

    Returns:
        financial_metrics dict from compute_core_metrics().
    """
    facts = build_facts_from_extraction(extraction)
    return compute_core_metrics(facts)


def build_facts_from_extraction(extraction: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Build the 'facts' dict for compute_core_metrics() from the
    structured output of ExtractionAgent.

    This function is the ONLY place where you map your parsed 10-K/iXBRL
    structure into the canonical metric names used by financial_calcs.py.

    It supports two patterns:

    1) If extraction["facts"] already exists and looks like:
          {
              "revenue": {period -> value},
              "net_income": {period -> value},
              ...
          }
       it is returned as-is.

    2) Otherwise, it will look for a more raw structure like
       extraction["ixbrl_facts"] or extraction["facts_raw"], and map
       common XBRL tags into the canonical names.
    """
    # Pattern 1: already normalized facts
    direct_facts = extraction.get("facts")
    if isinstance(direct_facts, dict) and direct_facts:
        return direct_facts  # assume caller already normalized keys & units

    # Pattern 2: build from ixbrl-style facts
    ix = (
        extraction.get("ixbrl_facts")
        or extraction.get("facts_raw")
        or extraction.get("raw_facts")
        or {}
    )

    facts: Dict[str, Dict[str, float]] = {}

    def _copy_metric(dst_key: str, *source_keys: str) -> None:
        """
        Copy from ix[source_key] into facts[dst_key].

        It uses the first source_key that exists and has a non-empty dict.

        Expected shape for ix[source_key]:
            {
                "2022-09-24": 394328.0,
                "2021-09-25": 365817.0,
                ...
            }
        """
        for tag in source_keys:
            src_map = ix.get(tag)
            if isinstance(src_map, dict) and src_map:
                facts[dst_key] = src_map
                return

    # 🔹 Map XBRL-ish tags into our canonical metric names.
    #    Adjust these tag names to match what your parser actually produces.
    #
    # Top-line revenue / net sales
    _copy_metric(
        "revenue",
        "NetSales",
        "RevenueFromContractWithCustomer",
        "SalesRevenueNet",
        "Revenues",
    )

    # Cost of sales / cost of goods sold
    _copy_metric(
        "cost_of_sales",
        "CostOfSales",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    )

    # Operating income
    _copy_metric(
        "operating_income",
        "OperatingIncomeLoss",
        "OperatingIncome",
    )

    # Net income
    _copy_metric(
        "net_income",
        "NetIncomeLoss",
        "NetIncome",
    )

    # Operating cash flow
    _copy_metric(
        "operating_cash_flow",
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    )

    # Capital expenditures (CapEx)
    _copy_metric(
        "capital_expenditures",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditures",
    )

    # Interest expense
    _copy_metric(
        "interest_expense",
        "InterestExpense",
        "InterestAndDebtExpense",
    )

    # Income tax expense / provision for income taxes
    _copy_metric(
        "income_tax_expense",
        "IncomeTaxExpenseBenefit",
        "ProvisionForIncomeTaxes",
    )

    # Depreciation & amortization
    _copy_metric(
        "depreciation_and_amortization",
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
    )

    return facts
