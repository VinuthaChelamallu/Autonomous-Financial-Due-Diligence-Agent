# tools/financial_calcs.py

"""
Core deterministic financial calculations.

This module is completely framework-agnostic.
It takes structured numeric inputs and returns
structured metrics, suitable for use by FinancialAgent
and MemoAgent.

Inputs:
    facts: dict in the shape produced by your extraction layer, e.g.:

        {
            "revenue": {
                "2022-09-24": 394328.0,
                "2021-09-25": 365817.0,
                "2020-09-26": 274515.0,
            },
            # or "net_sales" if you prefer that label
            "net_sales": {...},

            "cost_of_sales": {...},                # COGS
            "operating_income": {...},
            "net_income": {...},
            "operating_cash_flow": {...},
            "capital_expenditures": {...},

            # For EBITDA:
            "interest_expense": {...},
            "income_tax_expense": {...},
            "depreciation_and_amortization": {...},
        }

Output (example shape):

{
    "periods": ["2020-09-26", "2021-09-25", "2022-09-24"],
    "per_period": {
        "2022-09-24": {
            # base values
            "revenue": 394328.0,
            "net_income": 99803.0,
            "operating_cash_flow": 122151.0,
            "capital_expenditures": 10708.0,

            # derived metrics (margins as FRACTIONS, not %)
            "gross_profit": 170782.0,
            "gross_margin": 0.433,        # (revenue - cost_of_sales) / revenue
            "operating_income": 119437.0,
            "operating_margin": 0.303,    # operating_income / revenue
            "ebitda": 120000.0,
            "ebitda_margin": 0.304,       # ebitda / revenue
            "net_margin": 0.253,          # net_income / revenue
            "fcf": 111443.0,
            "fcf_margin": 0.283,          # fcf / revenue
        },
        ...
    },
    "yoy": {
        # growth rates also stored as FRACTIONS, not %:
        "revenue_growth": {
            "2021-09-25": 0.333,
            "2022-09-24": 0.078,
        },
        "net_income_growth": {
            ...
        },
        "fcf_growth": {
            ...
        },
        "ebitda_growth": {
            ...
        },
    },
}
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Iterable


def compute_core_metrics(facts: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Compute per-period financial metrics and simple YoY growth.

    Args:
        facts: dict of metric -> {period_end_date -> value}

    Returns:
        dict with "periods", "per_period", "yoy".
    """
    # 1) Collect all periods across all metrics
    periods = sorted(_collect_all_periods(facts))

    per_period: Dict[str, Dict[str, float]] = {}
    for period in periods:
        per_period[period] = _compute_metrics_for_period(facts, period)

    yoy = _compute_yoy_growth(per_period, periods)

    return {
        "periods": periods,
        "per_period": per_period,
        "yoy": yoy,
    }


# ---------------------------------------------------------------------------
# Helpers for reading input facts
# ---------------------------------------------------------------------------

def _collect_all_periods(facts: Dict[str, Dict[str, float]]) -> List[str]:
    periods = set()
    for metric_map in facts.values():
        if isinstance(metric_map, dict):
            periods.update(metric_map.keys())
    return list(periods)


def _get_value(
    facts: Dict[str, Dict[str, float]],
    metric: str,
    period: str,
) -> Optional[float]:
    metric_map = facts.get(metric) or {}
    return metric_map.get(period)


def _get_any_value(
    facts: Dict[str, Dict[str, float]],
    metric_names: Iterable[str],
    period: str,
) -> Optional[float]:
    """
    Helper to support alternative names, e.g. "net_sales" vs "revenue".
    Returns the first non-None value found.
    """
    for name in metric_names:
        val = _get_value(facts, name, period)
        if val is not None:
            return val
    return None


# ---------------------------------------------------------------------------
# Per-period metrics
# ---------------------------------------------------------------------------

def _compute_metrics_for_period(
    facts: Dict[str, Dict[str, float]],
    period: str,
) -> Dict[str, float]:
    """
    Compute metrics for a single period end date.

    All margin / growth metrics are stored as FRACTIONS (0–1), not percentages.
    """
    # Top-line: allow either "net_sales" or "revenue"
    revenue = _get_any_value(facts, ("net_sales", "revenue"), period)

    # Base values
    net_income = _get_value(facts, "net_income", period)
    op_cf = _get_value(facts, "operating_cash_flow", period)
    capex = _get_value(facts, "capital_expenditures", period)

    cost_of_sales = _get_value(facts, "cost_of_sales", period)
    operating_income = _get_value(facts, "operating_income", period)

    interest = _get_any_value(
        facts,
        ("interest_expense", "interest_and_other_expense", "interest_and_other"),
        period,
    )
    income_tax = _get_any_value(
        facts,
        ("income_tax_expense", "provision_for_income_taxes"),
        period,
    )
    d_and_a = _get_any_value(
        facts,
        ("depreciation_and_amortization", "depreciation_amortization"),
        period,
    )

    result: Dict[str, float] = {}

    # Store "revenue" key for downstream consistency
    if revenue is not None:
        result["revenue"] = revenue

    if net_income is not None:
        result["net_income"] = net_income
    if op_cf is not None:
        result["operating_cash_flow"] = op_cf
    if capex is not None:
        result["capital_expenditures"] = capex

    # 1) Gross margin: (Revenue – Cost of Sales) / Revenue
    if revenue is not None and cost_of_sales is not None:
        gross_profit = revenue - cost_of_sales
        result["gross_profit"] = gross_profit
        result["gross_margin"] = safe_div(gross_profit, revenue)

    # 2) Operating margin: Operating Income / Revenue
    if revenue is not None and operating_income is not None:
        result["operating_income"] = operating_income
        result["operating_margin"] = safe_div(operating_income, revenue)

    # 3) EBITDA & EBITDA margin:
    #    EBITDA = Net Income + Interest + Taxes + D&A
    if net_income is not None:
        ebitda_val = net_income
        if interest is not None:
            ebitda_val += interest
        if income_tax is not None:
            ebitda_val += income_tax
        if d_and_a is not None:
            ebitda_val += d_and_a

        result["ebitda"] = ebitda_val

        if revenue is not None:
            result["ebitda_margin"] = safe_div(ebitda_val, revenue)

    # 4) Net margin: Net Income / Revenue
    if revenue is not None and net_income is not None:
        result["net_margin"] = safe_div(net_income, revenue)

    # 5) Free Cash Flow & FCF margin: FCF = Op CF – CapEx
    if op_cf is not None and capex is not None:
        fcf = op_cf - capex
        result["fcf"] = fcf
        if revenue is not None:
            result["fcf_margin"] = safe_div(fcf, revenue)

    return result


# ---------------------------------------------------------------------------
# YoY growth
# ---------------------------------------------------------------------------

def _compute_yoy_growth(
    per_period: Dict[str, Dict[str, float]],
    periods: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute YoY growth for key metrics: revenue, net_income, fcf, ebitda.

    Returns dict: metric_growth_name -> { period -> growth_rate }
    where growth_rate = (current - previous) / previous
    (stored as a fraction, not a percentage).
    """
    metrics_to_track = [
        ("revenue", "revenue_growth"),
        ("net_income", "net_income_growth"),
        ("fcf", "fcf_growth"),
        ("ebitda", "ebitda_growth"),
    ]

    yoy: Dict[str, Dict[str, float]] = {
        growth_name: {} for _, growth_name in metrics_to_track
    }

    # Sort periods ascending so we can look back
    sorted_periods = sorted(periods)
    idx_by_period = {p: i for i, p in enumerate(sorted_periods)}

    for metric, growth_name in metrics_to_track:
        growth_map: Dict[str, float] = {}

        for period in sorted_periods:
            idx = idx_by_period[period]
            if idx == 0:
                # No prior period = no YoY
                continue

            prev_period = sorted_periods[idx - 1]

            current_val = per_period.get(period, {}).get(metric)
            prev_val = per_period.get(prev_period, {}).get(metric)

            if current_val is None or prev_val is None or prev_val == 0:
                continue

            growth = (current_val - prev_val) / prev_val
            growth_map[period] = growth

        yoy[growth_name] = growth_map

    return yoy


# ---------------------------------------------------------------------------
# Safe division
# ---------------------------------------------------------------------------

def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b
