"""Single-asset risk analytics and long-only portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .exceptions import InsufficientDataError, ValidationError
from .utils import validate_ticker

logger = logging.getLogger("equity_research.risk")


@dataclass(slots=True)
class PortfolioSolution:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float

    def frame(self) -> pd.DataFrame:
        return self.weights.rename("Weight").to_frame()


class RiskAnalyzer:
    def __init__(self, annualization: int = 252, risk_free_rate: float = 0.04):
        self.annualization = annualization
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        return prices.pct_change(fill_method=None).dropna(how="all")

    def performance_metrics(
        self,
        prices: pd.Series,
        benchmark_prices: pd.Series | None = None,
        confidence: float = 0.95,
    ) -> dict[str, float]:
        series = pd.to_numeric(prices, errors="coerce").dropna()
        returns = series.pct_change(fill_method=None).dropna()
        if len(returns) < 30:
            raise InsufficientDataError(
                "At least 30 return observations are required."
            )
        cumulative = (1 + returns).cumprod()
        years = len(returns) / self.annualization
        annual_return = cumulative.iloc[-1] ** (1 / years) - 1
        annual_volatility = returns.std(ddof=1) * np.sqrt(self.annualization)
        downside = returns.clip(upper=0)
        downside_deviation = (
            downside.std(ddof=1) * np.sqrt(self.annualization)
        )
        sharpe = (
            (annual_return - self.risk_free_rate) / annual_volatility
            if annual_volatility > 0
            else np.nan
        )
        sortino = (
            (annual_return - self.risk_free_rate) / downside_deviation
            if downside_deviation > 0
            else np.nan
        )
        drawdown = cumulative / cumulative.cummax() - 1
        historical_var = -float(returns.quantile(1 - confidence))
        tail = returns[returns <= returns.quantile(1 - confidence)]
        historical_cvar = -float(tail.mean()) if not tail.empty else np.nan

        metrics = {
            "annualized_return": float(annual_return),
            "annualized_volatility": float(annual_volatility),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(drawdown.min()),
            f"historical_var_{confidence:.0%}": historical_var,
            f"historical_cvar_{confidence:.0%}": historical_cvar,
            "best_day": float(returns.max()),
            "worst_day": float(returns.min()),
            "positive_day_ratio": float((returns > 0).mean()),
        }

        if benchmark_prices is not None:
            benchmark = pd.to_numeric(
                benchmark_prices, errors="coerce"
            ).dropna()
            aligned = pd.concat(
                [returns.rename("asset"), benchmark.pct_change(fill_method=None).rename("benchmark")],
                axis=1,
            ).dropna()
            if len(aligned) >= 30:
                covariance = aligned.cov().loc["asset", "benchmark"]
                benchmark_variance = aligned["benchmark"].var(ddof=1)
                beta = (
                    covariance / benchmark_variance
                    if benchmark_variance > 0
                    else np.nan
                )
                asset_mean = aligned["asset"].mean() * self.annualization
                benchmark_mean = (
                    aligned["benchmark"].mean() * self.annualization
                )
                alpha = asset_mean - (
                    self.risk_free_rate
                    + beta * (benchmark_mean - self.risk_free_rate)
                )
                correlation = aligned.corr().loc["asset", "benchmark"]
                metrics.update(
                    {
                        "beta": float(beta),
                        "capm_alpha": float(alpha),
                        "benchmark_correlation": float(correlation),
                    }
                )
        return metrics

    def drawdown_series(self, prices: pd.Series) -> pd.Series:
        series = pd.to_numeric(prices, errors="coerce").dropna()
        wealth = series / series.iloc[0]
        return (wealth / wealth.cummax() - 1).rename("Drawdown")


class PortfolioOptimizer:
    """Mean-variance optimizer with practical long-only constraints."""

    def __init__(
        self,
        annualization: int = 252,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.annualization = annualization
        self.risk_free_rate = risk_free_rate

    def _moments(
        self, prices: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        returns = prices.pct_change(fill_method=None).dropna(how="any")
        if len(returns) < 60:
            raise InsufficientDataError(
                "Portfolio optimization requires at least 60 complete return rows."
            )
        expected = returns.mean() * self.annualization
        covariance = returns.cov() * self.annualization
        return expected, covariance, returns

    @staticmethod
    def _portfolio_stats(
        weights: np.ndarray,
        expected: pd.Series,
        covariance: pd.DataFrame,
        risk_free_rate: float,
    ) -> tuple[float, float, float]:
        expected_return = float(weights @ expected.to_numpy())
        volatility = float(
            np.sqrt(weights @ covariance.to_numpy() @ weights)
        )
        sharpe = (
            (expected_return - risk_free_rate) / volatility
            if volatility > 0
            else np.nan
        )
        return expected_return, volatility, sharpe

    def maximize_sharpe(
        self,
        prices: pd.DataFrame,
        max_weight: float = 0.40,
    ) -> PortfolioSolution:
        expected, covariance, _ = self._moments(prices)
        n_assets = len(expected)
        if n_assets < 2:
            raise ValidationError("At least two assets are required.")
        if max_weight * n_assets < 1:
            raise ValidationError(
                "max_weight is too small for the number of assets."
            )
        initial = np.repeat(1 / n_assets, n_assets)
        bounds = [(0.0, max_weight)] * n_assets
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        def objective(weights: np.ndarray) -> float:
            _, volatility, sharpe = self._portfolio_stats(
                weights,
                expected,
                covariance,
                self.risk_free_rate,
            )
            return 1e6 if volatility <= 0 or not np.isfinite(sharpe) else -sharpe

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not result.success:
            raise ValidationError(
                f"Portfolio optimization failed: {result.message}"
            )
        exp_return, volatility, sharpe = self._portfolio_stats(
            result.x, expected, covariance, self.risk_free_rate
        )
        return PortfolioSolution(
            weights=pd.Series(result.x, index=expected.index),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
        )

    def minimum_variance(
        self,
        prices: pd.DataFrame,
        max_weight: float = 0.40,
    ) -> PortfolioSolution:
        expected, covariance, _ = self._moments(prices)
        n_assets = len(expected)
        if max_weight * n_assets < 1:
            raise ValidationError(
                "max_weight is too small for the number of assets."
            )
        initial = np.repeat(1 / n_assets, n_assets)
        bounds = [(0.0, max_weight)] * n_assets
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        def objective(weights: np.ndarray) -> float:
            return float(weights @ covariance.to_numpy() @ weights)

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not result.success:
            raise ValidationError(
                f"Portfolio optimization failed: {result.message}"
            )
        exp_return, volatility, sharpe = self._portfolio_stats(
            result.x, expected, covariance, self.risk_free_rate
        )
        return PortfolioSolution(
            weights=pd.Series(result.x, index=expected.index),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
        )

    def efficient_frontier(
        self,
        prices: pd.DataFrame,
        points: int = 30,
        max_weight: float = 0.40,
    ) -> pd.DataFrame:
        expected, covariance, _ = self._moments(prices)
        n_assets = len(expected)
        if max_weight * n_assets < 1:
            raise ValidationError(
                "max_weight is too small for the number of assets."
            )
        min_target = float(expected.min())
        max_target = float(expected.max())
        targets = np.linspace(min_target, max_target, points)
        initial = np.repeat(1 / n_assets, n_assets)
        bounds = [(0.0, max_weight)] * n_assets
        records: list[dict[str, float]] = []

        for target in targets:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {
                    "type": "eq",
                    "fun": lambda w, t=target: float(
                        w @ expected.to_numpy() - t
                    ),
                },
            ]

            result = minimize(
                lambda w: float(
                    w @ covariance.to_numpy() @ w
                ),
                initial,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if not result.success:
                continue
            ret, vol, sharpe = self._portfolio_stats(
                result.x, expected, covariance, self.risk_free_rate
            )
            records.append(
                {
                    "Expected Return": ret,
                    "Volatility": vol,
                    "Sharpe Ratio": sharpe,
                }
            )
        return pd.DataFrame(records).sort_values("Volatility")
