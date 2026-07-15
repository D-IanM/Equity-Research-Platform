"""Financial-statement normalization and ratio analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data import MarketDataBundle
from .utils import (
    divide_safe,
    latest_non_null,
    pct_change_safe,
    previous_non_null,
    safe_float,
)


LINE_ITEMS: dict[str, tuple[str, ...]] = {
    "revenue": ("TotalRevenue", "OperatingRevenue"),
    "cost_of_revenue": ("CostOfRevenue", "ReconciledCostOfRevenue"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncome",),
    "net_income": ("NetIncome", "NetIncomeCommonStockholders"),
    "ebit": ("EBIT", "OperatingIncome"),
    "ebitda": ("EBITDA", "NormalizedEBITDA"),
    "interest_expense": ("InterestExpense", "InterestExpenseNonOperating"),
    "tax_provision": ("TaxProvision",),
    "total_assets": ("TotalAssets",),
    "current_assets": ("CurrentAssets", "TotalCurrentAssets"),
    "cash": (
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalents",
        "CashFinancial",
    ),
    "inventory": ("Inventory",),
    "total_liabilities": (
        "TotalLiabilitiesNetMinorityInterest",
        "TotalLiabilities",
    ),
    "current_liabilities": ("CurrentLiabilities", "TotalCurrentLiabilities"),
    "equity": (
        "StockholdersEquity",
        "CommonStockEquity",
        "TotalEquityGrossMinorityInterest",
    ),
    "total_debt": ("TotalDebt",),
    "operating_cash_flow": ("OperatingCashFlow", "TotalCashFromOperatingActivities"),
    "capital_expenditure": ("CapitalExpenditure", "CapitalExpenditures"),
    "free_cash_flow": ("FreeCashFlow",),
    "shares": (
        "OrdinarySharesNumber",
        "ShareIssued",
    ),
}


@dataclass(slots=True)
class FundamentalSnapshot:
    ticker: str
    values: dict[str, float]
    ratios: dict[str, float]
    quality_score: float

    def to_dict(self) -> dict:
        return asdict(self)

    def ratio_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Metric": list(self.ratios), "Value": list(self.ratios.values())}
        ).set_index("Metric")


class FundamentalAnalyzer:
    """Extract consistent financial values and compute defensible ratios."""

    @staticmethod
    def _row(
        statement: pd.DataFrame,
        candidates: Iterable[str],
    ) -> pd.Series:
        if statement.empty:
            return pd.Series(dtype=float)
        normalized = {
            str(index).replace(" ", "").lower(): index for index in statement.index
        }
        for candidate in candidates:
            key = candidate.replace(" ", "").lower()
            if key in normalized:
                return pd.to_numeric(
                    statement.loc[normalized[key]], errors="coerce"
                )
        return pd.Series(dtype=float)

    def _value(
        self,
        statement: pd.DataFrame,
        key: str,
        previous: bool = False,
    ) -> float:
        row = self._row(statement, LINE_ITEMS[key])
        return previous_non_null(row) if previous else latest_non_null(row)

    def analyze(self, bundle: MarketDataBundle) -> FundamentalSnapshot:
        inc = bundle.income_statement
        bal = bundle.balance_sheet
        cash = bundle.cash_flow

        current: dict[str, float] = {}
        previous: dict[str, float] = {}

        income_keys = (
            "revenue",
            "cost_of_revenue",
            "gross_profit",
            "operating_income",
            "net_income",
            "ebit",
            "ebitda",
            "interest_expense",
            "tax_provision",
        )
        balance_keys = (
            "total_assets",
            "current_assets",
            "cash",
            "inventory",
            "total_liabilities",
            "current_liabilities",
            "equity",
            "total_debt",
            "shares",
        )
        cash_keys = ("operating_cash_flow", "capital_expenditure", "free_cash_flow")

        for key in income_keys:
            current[key] = self._value(inc, key)
            previous[key] = self._value(inc, key, previous=True)
        for key in balance_keys:
            current[key] = self._value(bal, key)
            previous[key] = self._value(bal, key, previous=True)
        for key in cash_keys:
            current[key] = self._value(cash, key)
            previous[key] = self._value(cash, key, previous=True)

        capex = current["capital_expenditure"]
        if not np.isfinite(current["free_cash_flow"]):
            current["free_cash_flow"] = (
                current["operating_cash_flow"] + capex
                if np.isfinite(capex) and capex < 0
                else current["operating_cash_flow"] - abs(capex)
            )
        prev_capex = previous["capital_expenditure"]
        if not np.isfinite(previous["free_cash_flow"]):
            previous["free_cash_flow"] = (
                previous["operating_cash_flow"] + prev_capex
                if np.isfinite(prev_capex) and prev_capex < 0
                else previous["operating_cash_flow"] - abs(prev_capex)
            )

        info = bundle.info
        market_cap = safe_float(info.get("marketCap"))
        enterprise_value = safe_float(info.get("enterpriseValue"))
        price = safe_float(info.get("currentPrice", info.get("regularMarketPrice")))
        shares = current["shares"]
        if not np.isfinite(shares):
            shares = safe_float(info.get("sharesOutstanding"))

        avg_assets = np.nanmean([current["total_assets"], previous["total_assets"]])
        avg_equity = np.nanmean([current["equity"], previous["equity"]])
        quick_assets = current["current_assets"] - max(current["inventory"], 0)

        ratios = {
            "revenue_growth": pct_change_safe(
                current["revenue"], previous["revenue"]
            ),
            "gross_margin": divide_safe(
                current["gross_profit"], current["revenue"]
            ),
            "operating_margin": divide_safe(
                current["operating_income"], current["revenue"]
            ),
            "net_margin": divide_safe(current["net_income"], current["revenue"]),
            "fcf_margin": divide_safe(
                current["free_cash_flow"], current["revenue"]
            ),
            "current_ratio": divide_safe(
                current["current_assets"], current["current_liabilities"]
            ),
            "quick_ratio": divide_safe(
                quick_assets, current["current_liabilities"]
            ),
            "debt_to_equity": divide_safe(
                current["total_debt"], current["equity"]
            ),
            "return_on_assets": divide_safe(current["net_income"], avg_assets),
            "return_on_equity": divide_safe(current["net_income"], avg_equity),
            "interest_coverage": divide_safe(
                current["ebit"], abs(current["interest_expense"])
            ),
            "operating_cash_conversion": divide_safe(
                current["operating_cash_flow"], current["net_income"]
            ),
            "price_to_earnings": safe_float(info.get("trailingPE")),
            "forward_pe": safe_float(info.get("forwardPE")),
            "price_to_sales": divide_safe(market_cap, current["revenue"]),
            "ev_to_ebitda": divide_safe(enterprise_value, current["ebitda"]),
            "free_cash_flow_yield": divide_safe(
                current["free_cash_flow"], market_cap
            ),
        }

        score_rules = [
            ratios["revenue_growth"] > 0,
            ratios["operating_margin"] > 0.10,
            ratios["net_margin"] > 0.08,
            ratios["fcf_margin"] > 0.08,
            ratios["current_ratio"] > 1.0,
            ratios["debt_to_equity"] < 1.5,
            ratios["return_on_equity"] > 0.12,
            ratios["interest_coverage"] > 3.0,
            ratios["operating_cash_conversion"] > 0.9,
            ratios["free_cash_flow_yield"] > 0.02,
        ]
        valid_rules = [bool(rule) for rule in score_rules if rule is not np.nan]
        quality_score = 10.0 * sum(valid_rules) / max(len(valid_rules), 1)

        values = {
            **current,
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "price": price,
            "shares": shares,
        }
        return FundamentalSnapshot(
            ticker=bundle.ticker,
            values=values,
            ratios=ratios,
            quality_score=quality_score,
        )
