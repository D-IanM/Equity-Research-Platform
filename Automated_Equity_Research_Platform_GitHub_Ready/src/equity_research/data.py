"""External data clients for market, SEC EDGAR, and FRED data."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Iterable

import pandas as pd
import requests

from .exceptions import DataSourceError, ValidationError
from .utils import (
    flatten_yfinance_columns,
    normalize_datetime_index,
    retry,
    validate_ticker,
)

logger = logging.getLogger("equity_research.data")


@dataclass(slots=True)
class MarketDataBundle:
    ticker: str
    prices: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)
    income_statement: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash_flow: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly_income_statement: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly_balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly_cash_flow: pd.DataFrame = field(default_factory=pd.DataFrame)
    actions: pd.DataFrame = field(default_factory=pd.DataFrame)


class YahooFinanceClient:
    """Research-oriented wrapper around yfinance with normalized outputs."""

    def __init__(self, timeout: int = 30, retries: int = 3) -> None:
        self.timeout = timeout
        self.retries = retries

    def _ticker(self, ticker: str):
        import yfinance as yf
        return yf.Ticker(validate_ticker(ticker))

    def history(
        self,
        ticker: str,
        start: str = "2016-01-01",
        end: str | None = None,
        interval: str = "1d",
        auto_adjust: bool = False,
    ) -> pd.DataFrame:
        import yfinance as yf
        symbol = validate_ticker(ticker)
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                frame = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=False,
                    threads=False,
                    timeout=self.timeout,
                    group_by="column",
                )
                frame = flatten_yfinance_columns(frame, symbol)
                frame = normalize_datetime_index(frame)
                if frame.empty:
                    raise DataSourceError(f"No price history returned for {symbol}.")
                required = {"Open", "High", "Low", "Close", "Volume"}
                missing = required.difference(frame.columns)
                if missing:
                    raise DataSourceError(
                        f"{symbol} history is missing columns: {sorted(missing)}"
                    )
                numeric_cols = frame.select_dtypes(include="number").columns
                frame[numeric_cols] = frame[numeric_cols].apply(
                    pd.to_numeric, errors="coerce"
                )
                return frame
            except Exception as exc:  # yfinance raises several inconsistent types
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.8 * (2 ** (attempt - 1)))
        raise DataSourceError(f"Could not download {symbol} history: {last_error}")

    def info(self, ticker: str) -> dict[str, Any]:
        symbol = validate_ticker(ticker)
        try:
            payload = self._ticker(symbol).get_info()
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.warning("Company metadata unavailable for %s: %s", symbol, exc)
            return {}

    @staticmethod
    def _statement(getter: Any, symbol: str, name: str) -> pd.DataFrame:
        try:
            frame = getter()
            if frame is None:
                return pd.DataFrame()
            frame = frame.copy()
            frame.columns = pd.to_datetime(frame.columns, errors="coerce")
            frame = frame.loc[:, ~frame.columns.isna()]
            frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
            return frame.sort_index(axis=1, ascending=False)
        except Exception as exc:
            logger.warning("%s unavailable for %s: %s", name, symbol, exc)
            return pd.DataFrame()

    def bundle(
        self,
        ticker: str,
        start: str = "2016-01-01",
        end: str | None = None,
    ) -> MarketDataBundle:
        symbol = validate_ticker(ticker)
        asset = self._ticker(symbol)

        try:
            actions = asset.actions.copy()
            if not actions.empty:
                actions = normalize_datetime_index(actions)
        except Exception:
            actions = pd.DataFrame()

        return MarketDataBundle(
            ticker=symbol,
            prices=self.history(symbol, start=start, end=end),
            info=self.info(symbol),
            income_statement=self._statement(
                lambda: asset.get_income_stmt(freq="yearly"),
                symbol,
                "annual income statement",
            ),
            balance_sheet=self._statement(
                lambda: asset.get_balance_sheet(freq="yearly"),
                symbol,
                "annual balance sheet",
            ),
            cash_flow=self._statement(
                lambda: asset.get_cash_flow(freq="yearly"),
                symbol,
                "annual cash-flow statement",
            ),
            quarterly_income_statement=self._statement(
                lambda: asset.get_income_stmt(freq="quarterly"),
                symbol,
                "quarterly income statement",
            ),
            quarterly_balance_sheet=self._statement(
                lambda: asset.get_balance_sheet(freq="quarterly"),
                symbol,
                "quarterly balance sheet",
            ),
            quarterly_cash_flow=self._statement(
                lambda: asset.get_cash_flow(freq="quarterly"),
                symbol,
                "quarterly cash-flow statement",
            ),
            actions=actions,
        )

    def close_matrix(
        self,
        tickers: Iterable[str],
        start: str = "2016-01-01",
        end: str | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        import yfinance as yf
        symbols = [validate_ticker(t) for t in dict.fromkeys(tickers)]
        if not symbols:
            raise ValidationError("At least one ticker is required.")
        try:
            frame = yf.download(
                symbols,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                progress=False,
                threads=True,
                timeout=self.timeout,
            )
            if frame.empty:
                raise DataSourceError("No multi-ticker price data returned.")
            if isinstance(frame.columns, pd.MultiIndex):
                if "Close" not in frame.columns.get_level_values(0):
                    raise DataSourceError("Close prices are absent from the download.")
                close = frame["Close"].copy()
            else:
                close = frame[["Close"]].rename(columns={"Close": symbols[0]})
            close = normalize_datetime_index(close)
            close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            return close
        except Exception as exc:
            raise DataSourceError(f"Could not download price matrix: {exc}") from exc


class SECClient:
    """Minimal SEC EDGAR JSON API client with a declared user agent."""

    TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
    COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(
        self,
        user_agent: str,
        timeout: int = 30,
        request_interval: float = 0.12,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValidationError(
                "SEC user agent must include a real name or organization and email."
            )
        self.timeout = timeout
        self.request_interval = request_interval
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov",
            }
        )
        self._ticker_map: dict[str, str] | None = None

    @retry(attempts=3)
    def _get_json(self, url: str, host: str = "data.sec.gov") -> dict[str, Any]:
        self.session.headers["Host"] = host
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        time.sleep(self.request_interval)
        payload = response.json()
        if not isinstance(payload, dict):
            raise DataSourceError(f"Unexpected SEC response from {url}")
        return payload

    def ticker_map(self, refresh: bool = False) -> dict[str, str]:
        if self._ticker_map is not None and not refresh:
            return self._ticker_map
        payload = self._get_json(self.TICKER_MAP_URL, host="www.sec.gov")
        mapping: dict[str, str] = {}
        for row in payload.values():
            ticker = str(row.get("ticker", "")).upper()
            cik = str(row.get("cik_str", "")).zfill(10)
            if ticker and cik:
                mapping[ticker] = cik
        self._ticker_map = mapping
        return mapping

    def cik_for_ticker(self, ticker: str) -> str:
        symbol = validate_ticker(ticker)
        try:
            return self.ticker_map()[symbol]
        except KeyError as exc:
            raise DataSourceError(f"No SEC CIK found for {symbol}.") from exc

    def company_facts(self, ticker: str) -> dict[str, Any]:
        cik = self.cik_for_ticker(ticker)
        return self._get_json(self.COMPANY_FACTS_URL.format(cik=cik))

    def submissions(self, ticker: str) -> dict[str, Any]:
        cik = self.cik_for_ticker(ticker)
        return self._get_json(self.SUBMISSIONS_URL.format(cik=cik))

    def recent_filings(
        self,
        ticker: str,
        forms: tuple[str, ...] = ("10-K", "10-Q", "8-K", "20-F", "6-K", "40-F"),
        limit: int = 20,
    ) -> pd.DataFrame:
        payload = self.submissions(ticker)
        recent = payload.get("filings", {}).get("recent", {})
        if not recent:
            return pd.DataFrame()
        frame = pd.DataFrame(recent)
        if "form" in frame:
            frame = frame[frame["form"].isin(forms)]
        keep = [
            col
            for col in (
                "filingDate",
                "reportDate",
                "form",
                "accessionNumber",
                "primaryDocument",
            )
            if col in frame.columns
        ]
        return frame[keep].head(limit).reset_index(drop=True)

    def fact_timeseries(
        self,
        ticker: str,
        concept: str,
        taxonomy: str = "us-gaap",
        unit: str | None = None,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
    ) -> pd.DataFrame:
        payload = self.company_facts(ticker)
        concept_payload = payload.get("facts", {}).get(taxonomy, {}).get(concept)
        if not concept_payload:
            return pd.DataFrame()
        units = concept_payload.get("units", {})
        selected_unit = unit or next(iter(units), None)
        if selected_unit is None:
            return pd.DataFrame()
        frame = pd.DataFrame(units.get(selected_unit, []))
        if frame.empty:
            return frame
        if "form" in frame:
            frame = frame[frame["form"].isin(forms)]
        for col in ("start", "end", "filed"):
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], errors="coerce")
        order = [col for col in ("end", "filed") if col in frame]
        if order:
            frame = frame.sort_values(order).drop_duplicates(
                subset=["end", "form"] if "form" in frame else ["end"],
                keep="last",
            )
        return frame.reset_index(drop=True)


class FredClient:
    """Small FRED series-observation client. A personal API key is required."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        if not api_key:
            raise ValidationError("A FRED API key is required.")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    @retry(attempts=3)
    def series(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end
        response = self.session.get(
            self.BASE_URL, params=params, timeout=self.timeout
        )
        response.raise_for_status()
        observations = response.json().get("observations", [])
        if not observations:
            raise DataSourceError(f"No FRED observations for {series_id}.")
        frame = pd.DataFrame(observations)
        index = pd.to_datetime(frame["date"], errors="coerce")
        values = pd.to_numeric(frame["value"], errors="coerce")
        output = pd.Series(values.to_numpy(), index=index, name=series_id)
        return output.loc[~output.index.isna()].sort_index()
