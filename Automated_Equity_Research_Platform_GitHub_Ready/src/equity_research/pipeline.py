"""End-to-end orchestration for one-company equity research."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data import FredClient, MarketDataBundle, SECClient, YahooFinanceClient
from .exceptions import EquityResearchError, InsufficientDataError
from .features import add_market_features
from .fundamentals import FundamentalAnalyzer, FundamentalSnapshot
from .reporting import ReportBuilder
from .risk import RiskAnalyzer
from .utils import save_frame, validate_ticker, write_json
from .valuation import (
    ComparableValuation,
    DCFInputs,
    DCFModel,
    DCFResult,
)

logger = logging.getLogger("equity_research.pipeline")


@dataclass(slots=True)
class ResearchResult:
    ticker: str
    bundle: MarketDataBundle
    features: pd.DataFrame
    fundamentals: FundamentalSnapshot
    risk_metrics: dict[str, float]
    dcf: DCFResult | None = None
    dcf_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_prices: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    peer_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparable_values: pd.DataFrame = field(default_factory=pd.DataFrame)
    sec_filings: pd.DataFrame = field(default_factory=pd.DataFrame)
    sec_company_facts: dict[str, Any] = field(default_factory=dict)
    macro_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "quality_score": self.fundamentals.quality_score,
            "fundamental_values": self.fundamentals.values,
            "fundamental_ratios": self.fundamentals.ratios,
            "risk_metrics": self.risk_metrics,
            "dcf": self.dcf.to_dict() if self.dcf else None,
        }


class EquityResearchPipeline:
    """A reproducible, research-oriented workflow for public equities."""

    DEFAULT_FRED_SERIES = {
        "FEDFUNDS": "Federal Funds Rate",
        "DGS10": "10-Year Treasury Yield",
        "CPIAUCSL": "Consumer Price Index",
        "UNRATE": "Unemployment Rate",
    }

    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or ProjectConfig()
        self.config.ensure_directories()
        self.yahoo = YahooFinanceClient(
            timeout=self.config.request_timeout,
            retries=self.config.request_retries,
        )
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.dcf_model = DCFModel()
        self.risk_analyzer = RiskAnalyzer(
            annualization=self.config.trading_days,
            risk_free_rate=self.config.risk_free_rate,
        )
        self.comparables = ComparableValuation(self.yahoo)

    def _sec_context(
        self, ticker: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not self.config.sec_user_agent:
            logger.info("SEC_USER_AGENT not set; SEC module skipped.")
            return pd.DataFrame(), {}
        try:
            client = SECClient(
                user_agent=self.config.sec_user_agent,
                timeout=self.config.request_timeout,
            )
            return client.recent_filings(ticker), client.company_facts(ticker)
        except EquityResearchError as exc:
            logger.warning("SEC data unavailable: %s", exc)
            return pd.DataFrame(), {}
        except Exception as exc:
            logger.warning("Unexpected SEC failure: %s", exc)
            return pd.DataFrame(), {}

    def _macro_context(self) -> pd.DataFrame:
        if not self.config.fred_api_key:
            logger.info("FRED_API_KEY not set; macro module skipped.")
            return pd.DataFrame()
        client = FredClient(
            self.config.fred_api_key,
            timeout=self.config.request_timeout,
        )
        series: list[pd.Series] = []
        for series_id, label in self.DEFAULT_FRED_SERIES.items():
            try:
                data = client.series(
                    series_id,
                    start=self.config.start_date,
                    end=self.config.end_date,
                )
                series.append(data.rename(label))
            except Exception as exc:
                logger.warning("FRED %s unavailable: %s", series_id, exc)
        return pd.concat(series, axis=1) if series else pd.DataFrame()

    def run(
        self,
        ticker: str,
        peers: Iterable[str] | None = None,
        dcf_inputs: DCFInputs | None = None,
        include_sec: bool = True,
        include_macro: bool = True,
        persist: bool = True,
    ) -> ResearchResult:
        symbol = validate_ticker(ticker)
        logger.info("Starting equity research pipeline for %s.", symbol)

        bundle = self.yahoo.bundle(
            symbol,
            start=self.config.start_date,
            end=self.config.end_date,
        )
        features = add_market_features(
            bundle.prices,
            annualization=self.config.trading_days,
        )
        fundamentals = self.fundamental_analyzer.analyze(bundle)

        benchmark_frame = self.yahoo.history(
            self.config.benchmark,
            start=self.config.start_date,
            end=self.config.end_date,
            auto_adjust=True,
        )
        benchmark_prices = benchmark_frame["Close"].rename(
            self.config.benchmark
        )
        price_column = "Adj Close" if "Adj Close" in bundle.prices else "Close"
        risk_metrics = self.risk_analyzer.performance_metrics(
            bundle.prices[price_column],
            benchmark_prices=benchmark_prices,
        )

        dcf_result: DCFResult | None = None
        sensitivity = pd.DataFrame()
        try:
            dcf_result = self.dcf_model.value(
                fundamentals, dcf_inputs or DCFInputs()
            )
            sensitivity = self.dcf_model.sensitivity(
                fundamentals,
                base_inputs=dcf_inputs or DCFInputs(),
            )
        except InsufficientDataError as exc:
            logger.warning("DCF not available for %s: %s", symbol, exc)

        peer_table = pd.DataFrame()
        implied_values = pd.DataFrame()
        peer_symbols = [
            validate_ticker(peer)
            for peer in (peers or [])
            if validate_ticker(peer) != symbol
        ]
        if peer_symbols:
            peer_table = self.comparables.peer_table(peer_symbols)
            try:
                implied_values = self.comparables.implied_values(
                    fundamentals, peer_table
                )
            except InsufficientDataError as exc:
                logger.warning("Comparable valuation unavailable: %s", exc)

        sec_filings, sec_facts = (
            self._sec_context(symbol) if include_sec else (pd.DataFrame(), {})
        )
        macro_data = (
            self._macro_context() if include_macro else pd.DataFrame()
        )

        result = ResearchResult(
            ticker=symbol,
            bundle=bundle,
            features=features,
            fundamentals=fundamentals,
            risk_metrics=risk_metrics,
            dcf=dcf_result,
            dcf_sensitivity=sensitivity,
            benchmark_prices=benchmark_prices,
            peer_table=peer_table,
            comparable_values=implied_values,
            sec_filings=sec_filings,
            sec_company_facts=sec_facts,
            macro_data=macro_data,
        )

        if persist:
            self.persist(result)
        logger.info("Completed equity research pipeline for %s.", symbol)
        return result

    def persist(self, result: ResearchResult) -> dict[str, Path]:
        symbol = result.ticker
        saved: dict[str, Path] = {}
        saved["prices"] = save_frame(
            result.bundle.prices,
            self.config.raw_dir / f"{symbol}_prices.parquet",
        )
        saved["features"] = save_frame(
            result.features,
            self.config.features_dir / f"{symbol}_market_features.parquet",
        )
        for name, frame in (
            ("income_statement", result.bundle.income_statement),
            ("balance_sheet", result.bundle.balance_sheet),
            ("cash_flow", result.bundle.cash_flow),
            ("quarterly_income", result.bundle.quarterly_income_statement),
            ("quarterly_balance", result.bundle.quarterly_balance_sheet),
            ("quarterly_cash_flow", result.bundle.quarterly_cash_flow),
            ("peer_table", result.peer_table),
            ("comparable_values", result.comparable_values),
            ("sec_filings", result.sec_filings),
            ("macro", result.macro_data),
            ("dcf_sensitivity", result.dcf_sensitivity),
        ):
            if not frame.empty:
                saved[name] = save_frame(
                    frame,
                    self.config.processed_dir / f"{symbol}_{name}.csv",
                )
        saved["summary"] = write_json(
            self.config.processed_dir / f"{symbol}_summary.json",
            result.summary(),
        )
        saved.update(
            {
                f"report_{key}": value
                for key, value in ReportBuilder().save(
                    result, self.config.reports_dir
                ).items()
            }
        )
        return saved
