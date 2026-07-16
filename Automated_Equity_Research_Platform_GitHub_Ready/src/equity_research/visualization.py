"""Reusable Plotly visualizations for notebooks and reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .valuation import DCFResult


def price_dashboard(prices: pd.DataFrame, ticker: str) -> go.Figure:
    frame = prices.copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.75, 0.25],
    )
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            name=ticker,
        ),
        row=1,
        col=1,
    )
    for column, label in (
        ("SMA20", "20D SMA"),
        ("SMA50", "50D SMA"),
        ("SMA200", "200D SMA"),
    ):
        if column in frame:
            fig.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame[column],
                    mode="lines",
                    name=label,
                ),
                row=1,
                col=1,
            )
    if "Volume" in frame:
        fig.add_trace(
            go.Bar(
                x=frame.index,
                y=frame["Volume"],
                name="Volume",
                opacity=0.6,
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        title=f"{ticker} Price and Trading Volume",
        template="plotly_white",
        height=720,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_y=1.03,
    )
    return fig


def drawdown_chart(drawdown: pd.Series, ticker: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill="tozeroy",
            name="Drawdown",
        )
    )
    fig.update_layout(
        title=f"{ticker} Drawdown",
        template="plotly_white",
        height=360,
        yaxis_tickformat=".0%",
        xaxis_title="Date",
    )
    return fig


def macro_series_chart(macro_data: pd.DataFrame) -> go.Figure:
    """Small-multiples chart of FRED macro series, each on its own y-axis scale."""
    columns = [c for c in macro_data.columns if macro_data[c].notna().any()]
    if not columns:
        return go.Figure()

    fig = make_subplots(
        rows=len(columns),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=columns,
    )
    for i, column in enumerate(columns, start=1):
        series = macro_data[column].dropna()
        fig.add_trace(
            go.Scatter(x=series.index, y=series, mode="lines", name=column),
            row=i,
            col=1,
        )
    fig.update_layout(
        template="plotly_white",
        height=max(600, 260 * len(columns)),
        showlegend=False,
        title="Macro Context (FRED)",
    )
    fig.update_xaxes(title_text="Date", row=len(columns), col=1)
    return fig


def ratio_chart(ratios: dict[str, float]) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Metric": list(ratios),
            "Value": list(ratios.values()),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    fig = px.bar(
        frame,
        x="Value",
        y="Metric",
        orientation="h",
        title="Fundamental Ratios",
    )
    fig.update_layout(
        template="plotly_white",
        height=max(450, 28 * len(frame)),
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


def dcf_projection_chart(result: DCFResult) -> go.Figure:
    frame = result.projection_frame().reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["Year"],
            y=frame["Projected FCF"],
            name="Projected FCF",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Year"],
            y=frame["Present Value"],
            mode="lines+markers",
            name="Present Value",
        )
    )
    fig.update_layout(
        title=f"{result.ticker} DCF Forecast",
        template="plotly_white",
        height=430,
        xaxis_title="Forecast Year",
        yaxis_title="Cash Flow",
    )
    return fig


def sensitivity_heatmap(table: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=table.to_numpy(dtype=float),
            x=table.columns,
            y=table.index,
            text=np.round(table.to_numpy(dtype=float), 2),
            texttemplate="$%{text}",
            hovertemplate=(
                "Terminal growth: %{x}<br>"
                "Discount rate: %{y}<br>"
                "Value/share: $%{z:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{ticker} DCF Sensitivity",
        template="plotly_white",
        height=430,
        xaxis_title="Terminal Growth",
        yaxis_title="Discount Rate",
    )
    return fig


def correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            text=np.round(corr.to_numpy(), 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Return Correlation Matrix",
        template="plotly_white",
        height=500,
    )
    return fig


def portfolio_weights_chart(weights: pd.Series, title: str) -> go.Figure:
    frame = weights.sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=frame.values,
            y=frame.index,
            orientation="h",
            text=[f"{x:.1%}" for x in frame.values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(350, 45 * len(frame)),
        xaxis_tickformat=".0%",
        xaxis_title="Portfolio Weight",
    )
    return fig


def efficient_frontier_chart(frontier: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        frontier,
        x="Volatility",
        y="Expected Return",
        color="Sharpe Ratio",
        title="Constrained Efficient Frontier",
    )
    fig.update_layout(
        template="plotly_white",
        height=450,
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
    )
    return fig
