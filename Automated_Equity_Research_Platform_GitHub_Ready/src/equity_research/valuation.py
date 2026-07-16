"""Discounted cash flow and trading-comparable valuation models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from .data import YahooFinanceClient
from .exceptions import InsufficientDataError, ValidationError
from .fundamentals import FundamentalSnapshot
from .utils import divide_safe, safe_float, validate_ticker

logger = logging.getLogger("equity_research.valuation")


@dataclass(slots=True)
class DCFInputs:
    forecast_years: int = 5
    initial_growth: float = 0.08
    terminal_growth: float = 0.025
    discount_rate: float = 0.09
    fade_growth: bool = True
    terminal_method: str = "gordon_growth"

    def validate(self) -> None:
        if not 1 <= self.forecast_years <= 15:
            raise ValidationError("forecast_years must be between 1 and 15.")
        if self.discount_rate <= self.terminal_growth:
            raise ValidationError(
                "discount_rate must be greater than terminal_growth."
            )
        if self.discount_rate <= -1 or self.terminal_growth <= -1:
            raise ValidationError("Growth and discount rates must exceed -100%.")
        if self.terminal_method != "gordon_growth":
            raise ValidationError("Only gordon_growth is implemented.")


@dataclass(slots=True)
class DCFResult:
    ticker: str
    base_free_cash_flow: float
    projected_cash_flows: list[float]
    projected_growth_rates: list[float]
    present_value_cash_flows: float
    terminal_value: float
    present_value_terminal: float
    enterprise_value: float
    equity_value: float
    intrinsic_value_per_share: float
    current_price: float
    upside_downside: float
    assumptions: DCFInputs

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["assumptions"] = asdict(self.assumptions)
        return payload

    def projection_frame(self) -> pd.DataFrame:
        years = np.arange(1, len(self.projected_cash_flows) + 1)
        discount_factors = 1 / (
            (1 + self.assumptions.discount_rate) ** years
        )
        return pd.DataFrame(
            {
                "Year": years,
                "Growth Rate": self.projected_growth_rates,
                "Projected FCF": self.projected_cash_flows,
                "Discount Factor": discount_factors,
                "Present Value": np.array(self.projected_cash_flows)
                * discount_factors,
            }
        ).set_index("Year")


class DCFModel:
    """Unlevered-style DCF using reported free cash flow as the starting base."""

    def value(
        self,
        snapshot: FundamentalSnapshot,
        assumptions: DCFInputs | None = None,
    ) -> DCFResult:
        inputs = assumptions or DCFInputs()
        inputs.validate()

        values = snapshot.values
        base_fcf = safe_float(values.get("free_cash_flow"))
        shares = safe_float(values.get("shares"))
        cash = safe_float(values.get("cash"), 0.0)
        debt = safe_float(values.get("total_debt"), 0.0)
        current_price = safe_float(values.get("price"))

        if not np.isfinite(base_fcf) or base_fcf <= 0:
            raise InsufficientDataError(
                "DCF requires a positive base free cash flow."
            )
        if not np.isfinite(shares) or shares <= 0:
            raise InsufficientDataError(
                "DCF requires a positive diluted or outstanding share count."
            )

        if inputs.fade_growth:
            growth_rates = np.linspace(
                inputs.initial_growth,
                inputs.terminal_growth,
                inputs.forecast_years,
            )
        else:
            growth_rates = np.repeat(
                inputs.initial_growth, inputs.forecast_years
            )

        projected: list[float] = []
        cash_flow = base_fcf
        for growth in growth_rates:
            cash_flow *= 1 + float(growth)
            projected.append(cash_flow)

        years = np.arange(1, inputs.forecast_years + 1)
        discount_factors = 1 / ((1 + inputs.discount_rate) ** years)
        pv_cash_flows = float(
            np.dot(np.array(projected), discount_factors)
        )

        terminal_value = (
            projected[-1]
            * (1 + inputs.terminal_growth)
            / (inputs.discount_rate - inputs.terminal_growth)
        )
        pv_terminal = terminal_value / (
            (1 + inputs.discount_rate) ** inputs.forecast_years
        )
        enterprise_value = pv_cash_flows + pv_terminal
        equity_value = enterprise_value + max(cash, 0) - max(debt, 0)
        intrinsic = equity_value / shares
        upside = (
            intrinsic / current_price - 1
            if np.isfinite(current_price) and current_price > 0
            else np.nan
        )

        return DCFResult(
            ticker=snapshot.ticker,
            base_free_cash_flow=base_fcf,
            projected_cash_flows=projected,
            projected_growth_rates=list(map(float, growth_rates)),
            present_value_cash_flows=pv_cash_flows,
            terminal_value=terminal_value,
            present_value_terminal=pv_terminal,
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            intrinsic_value_per_share=intrinsic,
            current_price=current_price,
            upside_downside=upside,
            assumptions=inputs,
        )

    def sensitivity(
        self,
        snapshot: FundamentalSnapshot,
        discount_rates: Iterable[float] | None = None,
        terminal_growth_rates: Iterable[float] | None = None,
        base_inputs: DCFInputs | None = None,
    ) -> pd.DataFrame:
        base = base_inputs or DCFInputs()
        rates = list(
            discount_rates
            if discount_rates is not None
            else np.arange(base.discount_rate - 0.02, base.discount_rate + 0.021, 0.01)
        )
        terminal_rates = list(
            terminal_growth_rates
            if terminal_growth_rates is not None
            else np.arange(
                max(0.0, base.terminal_growth - 0.01),
                base.terminal_growth + 0.011,
                0.005,
            )
        )
        table = pd.DataFrame(
            index=[f"{r:.1%}" for r in rates],
            columns=[f"{g:.1%}" for g in terminal_rates],
            dtype=float,
        )
        for rate in rates:
            for growth in terminal_rates:
                if rate <= growth:
                    table.loc[f"{rate:.1%}", f"{growth:.1%}"] = np.nan
                    continue
                inputs = DCFInputs(
                    forecast_years=base.forecast_years,
                    initial_growth=base.initial_growth,
                    terminal_growth=float(growth),
                    discount_rate=float(rate),
                    fade_growth=base.fade_growth,
                )
                try:
                    table.loc[f"{rate:.1%}", f"{growth:.1%}"] = (
                        self.value(snapshot, inputs).intrinsic_value_per_share
                    )
                except (ValidationError, InsufficientDataError):
                    table.loc[f"{rate:.1%}", f"{growth:.1%}"] = np.nan
        table.index.name = "Discount Rate"
        table.columns.name = "Terminal Growth"
        return table


class ComparableValuation:
    """Peer multiple collection and simple implied-price analysis."""

    def __init__(self, yahoo: YahooFinanceClient | None = None) -> None:
        self.yahoo = yahoo or YahooFinanceClient()

    def peer_table(self, tickers: Iterable[str]) -> pd.DataFrame:
        records: list[dict[str, float | str]] = []
        for raw_symbol in dict.fromkeys(tickers):
            symbol = validate_ticker(raw_symbol)
            info = self.yahoo.info(symbol)
            if not info:
                logger.warning("Skipping %s because metadata is unavailable.", symbol)
                continue
            records.append(
                {
                    "Ticker": symbol,
                    "Price": safe_float(
                        info.get("currentPrice", info.get("regularMarketPrice"))
                    ),
                    "Market Cap": safe_float(info.get("marketCap")),
                    "Enterprise Value": safe_float(info.get("enterpriseValue")),
                    "Revenue": safe_float(info.get("totalRevenue")),
                    "EBITDA": safe_float(info.get("ebitda")),
                    "Net Income": safe_float(
                        info.get("netIncomeToCommon")
                    ),
                    "Trailing P/E": safe_float(info.get("trailingPE")),
                    "Forward P/E": safe_float(info.get("forwardPE")),
                    "Price/Sales": safe_float(info.get("priceToSalesTrailing12Months")),
                    "EV/EBITDA": safe_float(info.get("enterpriseToEbitda")),
                    "EV/Revenue": safe_float(info.get("enterpriseToRevenue")),
                }
            )
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records).set_index("Ticker")

    def implied_values(
        self,
        target: FundamentalSnapshot,
        peers: pd.DataFrame,
    ) -> pd.DataFrame:
        if peers.empty:
            return pd.DataFrame()
        values = target.values
        current_price = safe_float(values.get("price"))
        shares = safe_float(values.get("shares"))
        net_income = safe_float(values.get("net_income"))
        revenue = safe_float(values.get("revenue"))
        ebitda = safe_float(values.get("ebitda"))
        debt = safe_float(values.get("total_debt"), 0.0)
        cash = safe_float(values.get("cash"), 0.0)

        if not np.isfinite(shares) or shares <= 0:
            raise InsufficientDataError(
                "Comparable valuation requires a positive share count."
            )

        median_pe = safe_float(peers["Trailing P/E"].replace([np.inf, -np.inf], np.nan).median())
        median_ps = safe_float(peers["Price/Sales"].replace([np.inf, -np.inf], np.nan).median())
        median_ev_ebitda = safe_float(peers["EV/EBITDA"].replace([np.inf, -np.inf], np.nan).median())

        implied_pe = divide_safe(net_income * median_pe, shares)
        implied_ps = divide_safe(revenue * median_ps, shares)
        implied_ev_ebitda = divide_safe(
            (ebitda * median_ev_ebitda) - debt + cash,
            shares,
        )

        result = pd.DataFrame(
            {
                "Method": ["Peer P/E", "Peer Price/Sales", "Peer EV/EBITDA"],
                "Peer Median Multiple": [median_pe, median_ps, median_ev_ebitda],
                "Implied Price": [implied_pe, implied_ps, implied_ev_ebitda],
            }
        ).set_index("Method")
        result["Current Price"] = current_price
        result["Upside/Downside"] = (
            result["Implied Price"] / current_price - 1
            if np.isfinite(current_price) and current_price > 0
            else np.nan
        )
        return result
