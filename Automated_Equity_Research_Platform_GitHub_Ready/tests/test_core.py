from __future__ import annotations

import numpy as np
import pandas as pd

from equity_research.data import MarketDataBundle
from equity_research.features import add_market_features
from equity_research.fundamentals import FundamentalAnalyzer
from equity_research.risk import PortfolioOptimizer, RiskAnalyzer
from equity_research.valuation import DCFInputs, DCFModel


def synthetic_prices(rows: int = 600, assets: int = 1):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-01", periods=rows)
    returns = rng.normal(0.0004, 0.012, size=(rows, assets))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    if assets == 1:
        close = pd.Series(prices[:, 0], index=dates)
        return pd.DataFrame(
            {
                "Open": close * 0.998,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Adj Close": close,
                "Volume": 1_000_000,
            }
        )
    return pd.DataFrame(
        prices,
        index=dates,
        columns=[f"A{i}" for i in range(assets)],
    )


def synthetic_bundle() -> MarketDataBundle:
    columns = pd.to_datetime(["2025-12-31", "2024-12-31"])
    income = pd.DataFrame(
        {
            columns[0]: [
                1200, 500, 700, 300, 220, 300, 360, 20, 50
            ],
            columns[1]: [
                1000, 450, 550, 240, 180, 240, 300, 20, 40
            ],
        },
        index=[
            "TotalRevenue",
            "CostOfRevenue",
            "GrossProfit",
            "OperatingIncome",
            "NetIncome",
            "EBIT",
            "EBITDA",
            "InterestExpense",
            "TaxProvision",
        ],
    )
    balance = pd.DataFrame(
        {
            columns[0]: [2000, 900, 450, 80, 900, 400, 1100, 300, 100],
            columns[1]: [1800, 800, 380, 70, 850, 390, 950, 350, 100],
        },
        index=[
            "TotalAssets",
            "CurrentAssets",
            "CashCashEquivalentsAndShortTermInvestments",
            "Inventory",
            "TotalLiabilities",
            "CurrentLiabilities",
            "StockholdersEquity",
            "TotalDebt",
            "OrdinarySharesNumber",
        ],
    )
    cash_flow = pd.DataFrame(
        {
            columns[0]: [300, -80, 220],
            columns[1]: [260, -70, 190],
        },
        index=[
            "OperatingCashFlow",
            "CapitalExpenditure",
            "FreeCashFlow",
        ],
    )
    return MarketDataBundle(
        ticker="TEST",
        prices=synthetic_prices(),
        info={
            "marketCap": 2500,
            "enterpriseValue": 2350,
            "currentPrice": 25,
            "sharesOutstanding": 100,
            "trailingPE": 11.36,
            "forwardPE": 10.0,
        },
        income_statement=income,
        balance_sheet=balance,
        cash_flow=cash_flow,
    )


def test_market_features_have_expected_columns():
    features = add_market_features(synthetic_prices())
    required = {
        "Return",
        "LogReturn",
        "SMA20",
        "SMA50",
        "SMA200",
        "RollingVolatility",
        "RSI14",
        "Drawdown",
    }
    assert required.issubset(features.columns)
    assert features["Drawdown"].max() <= 1e-12


def test_fundamental_analyzer():
    snapshot = FundamentalAnalyzer().analyze(synthetic_bundle())
    assert np.isclose(snapshot.ratios["revenue_growth"], 0.20)
    assert np.isclose(snapshot.ratios["gross_margin"], 700 / 1200)
    assert np.isclose(snapshot.ratios["current_ratio"], 900 / 400)
    assert 0 <= snapshot.quality_score <= 10


def test_dcf_produces_positive_intrinsic_value():
    snapshot = FundamentalAnalyzer().analyze(synthetic_bundle())
    result = DCFModel().value(
        snapshot,
        DCFInputs(
            initial_growth=0.06,
            discount_rate=0.10,
            terminal_growth=0.025,
        ),
    )
    assert result.intrinsic_value_per_share > 0
    assert len(result.projected_cash_flows) == 5


def test_risk_metrics_and_drawdown():
    prices = synthetic_prices()["Adj Close"]
    benchmark = synthetic_prices()["Adj Close"] * 1.001
    analyzer = RiskAnalyzer(risk_free_rate=0.03)
    metrics = analyzer.performance_metrics(prices, benchmark)
    assert "sharpe_ratio" in metrics
    assert "beta" in metrics
    assert metrics["max_drawdown"] <= 0


def test_portfolio_optimizer_weights_sum_to_one():
    prices = synthetic_prices(assets=4)
    optimizer = PortfolioOptimizer(risk_free_rate=0.03)
    solution = optimizer.maximize_sharpe(prices, max_weight=0.40)
    assert np.isclose(solution.weights.sum(), 1.0, atol=1e-6)
    assert (solution.weights >= -1e-8).all()
    assert (solution.weights <= 0.400001).all()
