"""Streamlit web app for the Automated Equity Research Platform.

Run locally:
    streamlit run app.py

Deploy:
    Push this repository to GitHub and deploy app.py on Streamlit Community Cloud.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_research import ProjectConfig, EquityResearchPipeline
from equity_research.valuation import DCFInputs
from equity_research.visualization import (
    dcf_projection_chart,
    drawdown_chart,
    price_dashboard,
    ratio_chart,
    sensitivity_heatmap,
    correlation_heatmap,
    portfolio_weights_chart,
    efficient_frontier_chart,
)
from equity_research.risk import PortfolioOptimizer


st.set_page_config(
    page_title="Automated Equity Research Platform",
    page_icon="📈",
    layout="wide",
)


def load_secrets() -> None:
    """Load optional secrets from Streamlit Cloud or environment variables."""
    for name in ("SEC_USER_AGENT", "FRED_API_KEY"):
        try:
            value = st.secrets.get(name, None)
        except Exception:
            value = None
        if value and not os.getenv(name):
            os.environ[name] = str(value)


@st.cache_resource(show_spinner=False)
def get_pipeline(start_date: str, benchmark: str, risk_free_rate: float) -> EquityResearchPipeline:
    config = ProjectConfig(
        project_root=PROJECT_ROOT,
        start_date=start_date,
        benchmark=benchmark,
        risk_free_rate=risk_free_rate,
    )
    return EquityResearchPipeline(config)


@st.cache_data(show_spinner=False, ttl=1800)
def run_research_cached(
    ticker: str,
    peers_tuple: tuple[str, ...],
    fcf_growth: float,
    discount_rate: float,
    terminal_growth: float,
    start_date: str,
    benchmark: str,
    risk_free_rate: float,
    include_sec: bool,
    include_fred: bool,
):
    pipeline = get_pipeline(start_date, benchmark, risk_free_rate)
    result = pipeline.run(
        ticker=ticker,
        peers=list(peers_tuple),
        dcf_inputs=DCFInputs(
            forecast_years=5,
            initial_growth=fcf_growth,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
        ),
        include_sec=include_sec,
        include_macro=include_fred,
        persist=True,
    )
    return result


def format_percent(value: float) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.2%}"


def format_number(value: float) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:,.2f}"


def format_money(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    if abs(value) >= 1e12:
        return f"${value / 1e12:,.2f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.2f}M"
    return f"${value:,.2f}"


load_secrets()

st.title("Automated Equity Research Platform")
st.caption(
    "Analyze public companies through data collection, financial ratios, valuation, risk analytics, peer comparison, and automated reports."
)
st.warning(
    "Educational research only. This is not financial advice or a recommendation to buy or sell securities.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Research Inputs")

    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    peers_raw = st.text_input(
        "Peer tickers",
        value="MSFT,GOOGL,AMZN,META",
        help="Comma-separated public-company tickers.",
    )
    peers = tuple(
        dict.fromkeys(
            peer.strip().upper()
            for peer in peers_raw.split(",")
            if peer.strip() and peer.strip().upper() != ticker
        )
    )

    st.subheader("DCF assumptions")
    fcf_growth = st.slider(
        "FCF growth",
        min_value=-0.05,
        max_value=0.25,
        value=0.08,
        step=0.005,
        format="%.3f",
        help="Expected annual free-cash-flow growth during the explicit forecast period.",
    )
    discount_rate = st.slider(
        "Discount rate",
        min_value=0.06,
        max_value=0.18,
        value=0.09,
        step=0.005,
        format="%.3f",
        help="Required return / risk rate used to discount future cash flows.",
    )
    terminal_growth = st.slider(
        "Terminal growth",
        min_value=0.00,
        max_value=0.05,
        value=0.025,
        step=0.005,
        format="%.3f",
        help="Long-term growth after the explicit forecast period.",
    )

    st.subheader("Settings")
    start_date = st.text_input("Start date", value="2018-01-01")
    benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
    risk_free_rate = st.slider(
        "Risk-free rate",
        min_value=0.00,
        max_value=0.10,
        value=0.04,
        step=0.005,
        format="%.3f",
    )

    sec_available = bool(os.getenv("SEC_USER_AGENT"))
    fred_available = bool(os.getenv("FRED_API_KEY"))
    include_sec = st.checkbox("Include SEC", value=False, disabled=not sec_available)
    include_fred = st.checkbox("Include FRED", value=False, disabled=not fred_available)

    if not sec_available:
        st.caption("SEC disabled: add SEC_USER_AGENT in Streamlit secrets.")
    if not fred_available:
        st.caption("FRED disabled: add FRED_API_KEY in Streamlit secrets.")

    run_button = st.button("Run Research", type="primary", use_container_width=True)

st.markdown(
    """
    **How to use:** enter a ticker, add peers, choose assumptions, then click **Run Research**.
    Reports will be generated in the `reports/` folder and downloadable from this page.
    """
)

if not run_button:
    st.info("Enter inputs in the sidebar and click **Run Research** to begin.")
    st.stop()

if not ticker:
    st.error("Please enter a ticker.")
    st.stop()

if discount_rate <= terminal_growth:
    st.error("Discount rate must be greater than terminal growth.")
    st.stop()

try:
    with st.spinner(f"Running research pipeline for {ticker}..."):
        result = run_research_cached(
            ticker=ticker,
            peers_tuple=peers,
            fcf_growth=fcf_growth,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            start_date=start_date,
            benchmark=benchmark,
            risk_free_rate=risk_free_rate,
            include_sec=include_sec,
            include_fred=include_fred,
        )
except Exception as exc:
    st.error(f"{type(exc).__name__}: {exc}")
    st.stop()

company_name = result.bundle.info.get("longName", result.ticker)
st.header(f"{company_name} ({result.ticker})")

values = result.fundamentals.values
risk_metrics = result.risk_metrics

col1, col2, col3, col4 = st.columns(4)
col1.metric("Quality score", f"{result.fundamentals.quality_score:.1f}/10")
col2.metric("Current price", format_money(values.get("price", np.nan)))
if result.dcf:
    col3.metric("DCF value/share", format_money(result.dcf.intrinsic_value_per_share))
    col4.metric("DCF upside/downside", format_percent(result.dcf.upside_downside))
else:
    col3.metric("DCF value/share", "N/A")
    col4.metric("DCF upside/downside", "N/A")

tabs = st.tabs(
    [
        "Overview",
        "Market Chart",
        "Fundamentals",
        "Valuation",
        "Risk",
        "Peers",
        "Report",
        "Model Notes",
    ]
)

with tabs[0]:
    st.subheader("Company overview")
    st.write(result.bundle.info.get("longBusinessSummary", "No company summary available."))
    overview = pd.Series(
        {
            "Sector": result.bundle.info.get("sector", "N/A"),
            "Industry": result.bundle.info.get("industry", "N/A"),
            "Market Cap": format_money(values.get("market_cap", np.nan)),
            "Enterprise Value": format_money(values.get("enterprise_value", np.nan)),
            "Revenue": format_money(values.get("revenue", np.nan)),
            "Net Income": format_money(values.get("net_income", np.nan)),
            "Free Cash Flow": format_money(values.get("free_cash_flow", np.nan)),
            "Cash": format_money(values.get("cash", np.nan)),
            "Total Debt": format_money(values.get("total_debt", np.nan)),
        }
    )
    st.dataframe(overview.to_frame("Value"), use_container_width=True)

with tabs[1]:
    st.subheader("Price and volume")
    st.plotly_chart(price_dashboard(result.features.tail(1250), result.ticker), use_container_width=True)
    price_column = "Adj Close" if "Adj Close" in result.bundle.prices else "Close"
    st.plotly_chart(
        drawdown_chart(
            get_pipeline(start_date, benchmark, risk_free_rate).risk_analyzer.drawdown_series(
                result.bundle.prices[price_column]
            ),
            result.ticker,
        ),
        use_container_width=True,
    )

with tabs[2]:
    st.subheader("Fundamental analysis")
    st.plotly_chart(ratio_chart(result.fundamentals.ratios), use_container_width=True)
    ratio_frame = result.fundamentals.ratio_frame()
    st.dataframe(ratio_frame, use_container_width=True)

with tabs[3]:
    st.subheader("Valuation")
    if result.dcf:
        dcf_table = pd.Series(
            {
                "Base FCF": result.dcf.base_free_cash_flow,
                "PV of Forecast Cash Flows": result.dcf.present_value_cash_flows,
                "PV of Terminal Value": result.dcf.present_value_terminal,
                "Enterprise Value": result.dcf.enterprise_value,
                "Equity Value": result.dcf.equity_value,
                "Intrinsic Value / Share": result.dcf.intrinsic_value_per_share,
                "Current Price": result.dcf.current_price,
                "Upside / Downside": result.dcf.upside_downside,
            }
        ).to_frame("Value")
        st.dataframe(dcf_table, use_container_width=True)
        st.plotly_chart(dcf_projection_chart(result.dcf), use_container_width=True)
        if not result.dcf_sensitivity.empty:
            st.plotly_chart(sensitivity_heatmap(result.dcf_sensitivity, result.ticker), use_container_width=True)
    else:
        st.warning(
            "DCF was skipped because the available data did not provide positive base free cash flow or required share data."
        )

with tabs[4]:
    st.subheader("Risk analytics")
    st.dataframe(pd.Series(risk_metrics, name="Value").to_frame(), use_container_width=True)
    if peers:
        tickers_for_portfolio = tuple(dict.fromkeys((result.ticker,) + peers))
        try:
            close = get_pipeline(start_date, benchmark, risk_free_rate).yahoo.close_matrix(
                list(tickers_for_portfolio),
                start=start_date,
                auto_adjust=True,
            )
            returns = close.pct_change(fill_method=None).dropna()
            st.plotly_chart(correlation_heatmap(returns), use_container_width=True)

            optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
            asset_prices = close.dropna()
            max_weight = max(0.40, 1 / len(asset_prices.columns))
            max_sharpe = optimizer.maximize_sharpe(asset_prices, max_weight=max_weight)
            min_variance = optimizer.minimum_variance(asset_prices, max_weight=max_weight)
            st.plotly_chart(
                portfolio_weights_chart(max_sharpe.weights, "Maximum-Sharpe Portfolio"),
                use_container_width=True,
            )
            st.plotly_chart(
                portfolio_weights_chart(min_variance.weights, "Minimum-Variance Portfolio"),
                use_container_width=True,
            )
            frontier = optimizer.efficient_frontier(asset_prices, max_weight=max_weight)
            if not frontier.empty:
                st.plotly_chart(efficient_frontier_chart(frontier), use_container_width=True)
        except Exception as exc:
            st.info(f"Portfolio optimization skipped: {exc}")
    else:
        st.info("Add peers to run correlation and portfolio optimization.")

with tabs[5]:
    st.subheader("Peer comparison")
    if not result.peer_table.empty:
        st.dataframe(result.peer_table, use_container_width=True)
    else:
        st.info("No peer data available.")
    st.subheader("Comparable valuation")
    if not result.comparable_values.empty:
        st.dataframe(result.comparable_values, use_container_width=True)
    else:
        st.info("Comparable valuation unavailable.")

with tabs[6]:
    st.subheader("Generated report")
    markdown_path = PROJECT_ROOT / "reports" / f"{result.ticker}_equity_research.md"
    html_path = PROJECT_ROOT / "reports" / f"{result.ticker}_equity_research.html"

    if markdown_path.exists():
        markdown_text = markdown_path.read_text(encoding="utf-8")
        st.download_button(
            "Download Markdown Report",
            data=markdown_text,
            file_name=markdown_path.name,
            mime="text/markdown",
        )
        with st.expander("Preview Markdown Report"):
            st.markdown(markdown_text)
    else:
        st.warning("Markdown report was not found.")

    if html_path.exists():
        st.download_button(
            "Download HTML Report",
            data=html_path.read_text(encoding="utf-8"),
            file_name=html_path.name,
            mime="text/html",
        )

with tabs[7]:
    st.subheader("Model notes and limitations")
    st.markdown(
        """
        - Yahoo Finance data can be incomplete or inconsistent.
        - Company statement classifications can vary.
        - The DCF uses simplified assumptions and is highly sensitive to growth and discount rates.
        - If free cash flow is negative, the DCF may be skipped instead of forcing a misleading output.
        - Peer selection can materially change comparable valuation.
        - Historical risk does not predict future risk.
        - Portfolio optimization is sensitive to historical returns and covariance.
        - The quality score is transparent and rules-based, not a trained prediction model.
        """
    )
