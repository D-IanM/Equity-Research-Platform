"""Markdown and HTML equity research report generation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import sanitize_for_json


def _money(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    magnitude = abs(value)
    if magnitude >= 1e12:
        return f"${value / 1e12:,.2f}T"
    if magnitude >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"${value / 1e6:,.2f}M"
    return f"${value:,.2f}"


def _number(value: float) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:,.2f}"


def _percent(value: float) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.1%}"


def _signal(quality_score: float, upside: float) -> tuple[str, str]:
    """Return a transparent research label, not personalized investment advice."""
    if np.isfinite(upside) and quality_score >= 7 and upside >= 0.20:
        return "Favorable", "Strong quality score and material modeled valuation upside."
    if np.isfinite(upside) and (upside <= -0.20 or quality_score < 4):
        return "Cautious", "Weak quality and/or modeled valuation downside warrants caution."
    return "Neutral", "Mixed evidence or limited valuation margin of safety."


class ReportBuilder:
    """Create portable reports that can be committed to GitHub."""

    def markdown(self, result: Any) -> str:
        snapshot = result.fundamentals
        dcf = result.dcf
        ratios = snapshot.ratios
        values = snapshot.values
        upside = dcf.upside_downside if dcf is not None else np.nan
        signal, rationale = _signal(snapshot.quality_score, upside)
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        ratio_lines = "\n".join(
            f"| {name.replace('_', ' ').title()} | "
            f"{_percent(value) if any(token in name for token in ('margin', 'growth', 'return', 'yield')) else _number(value)} |"
            for name, value in ratios.items()
        )

        risk_lines = "\n".join(
            f"| {name.replace('_', ' ').title()} | "
            f"{_percent(value) if any(token in name for token in ('return', 'volatility', 'drawdown', 'var_', 'cvar_', 'alpha', 'ratio')) and 'sharpe' not in name and 'sortino' not in name else _number(value)} |"
            for name, value in result.risk_metrics.items()
        )

        dcf_section = "DCF could not be completed with the available data."
        if dcf is not None:
            dcf_section = f"""
| Metric | Value |
|---|---:|
| Current Price | {_money(dcf.current_price)} |
| Intrinsic Value / Share | {_money(dcf.intrinsic_value_per_share)} |
| Modeled Upside / Downside | {_percent(dcf.upside_downside)} |
| Enterprise Value | {_money(dcf.enterprise_value)} |
| Equity Value | {_money(dcf.equity_value)} |
| Discount Rate | {_percent(dcf.assumptions.discount_rate)} |
| Terminal Growth | {_percent(dcf.assumptions.terminal_growth)} |
"""

        company = result.bundle.info.get("longName", result.ticker)
        sector = result.bundle.info.get("sector", "N/A")
        industry = result.bundle.info.get("industry", "N/A")
        summary = result.bundle.info.get("longBusinessSummary", "No summary available.")

        return f"""# {company} ({result.ticker}) Equity Research Report

**Generated:** {generated}  
**Research Signal:** **{signal}** — {rationale}  
**Quality Score:** {snapshot.quality_score:.1f}/10

> Educational research output only. This is not personalized investment advice, a solicitation, or a guarantee of future performance.

## 1. Company Overview

- **Sector:** {sector}
- **Industry:** {industry}
- **Market Capitalization:** {_money(values.get("market_cap", np.nan))}
- **Enterprise Value:** {_money(values.get("enterprise_value", np.nan))}
- **Current Price:** {_money(values.get("price", np.nan))}

{summary}

## 2. Financial Snapshot

| Metric | Value |
|---|---:|
| Revenue | {_money(values.get("revenue", np.nan))} |
| Operating Income | {_money(values.get("operating_income", np.nan))} |
| Net Income | {_money(values.get("net_income", np.nan))} |
| Free Cash Flow | {_money(values.get("free_cash_flow", np.nan))} |
| Cash | {_money(values.get("cash", np.nan))} |
| Total Debt | {_money(values.get("total_debt", np.nan))} |

## 3. Fundamental Ratios

| Metric | Value |
|---|---:|
{ratio_lines}

## 4. Discounted Cash Flow Valuation

{dcf_section}

## 5. Market Risk and Performance

| Metric | Value |
|---|---:|
{risk_lines}

## 6. Peer Comparison

{result.peer_table.round(2).to_markdown() if not result.peer_table.empty else "No peer table was requested or available."}

## 7. Comparable Valuation

{result.comparable_values.round(2).to_markdown() if not result.comparable_values.empty else "Comparable implied values were unavailable."}

## 8. SEC Filing Context

{result.sec_filings.to_markdown(index=False) if not result.sec_filings.empty else "SEC integration was not configured or no matching filings were returned."}

## 9. Methodology and Limitations

- Market and company data are collected programmatically and can contain delays, restatements, missing fields, or provider inconsistencies.
- The DCF is assumption-sensitive. Review the sensitivity table rather than treating one point estimate as certain.
- Historical risk metrics do not predict future risk.
- The quality score is a transparent rules-based summary, not a trained credit or investment-rating model.
- This project is intended for research, education, and portfolio demonstration.

## 10. Reproducibility

The report was generated by the `Automated Equity Research Platform` package from the accompanying Colab notebooks. Inputs, source code, and calculation methods are included in the repository.
"""

    def html(self, result: Any) -> str:
        markdown_text = self.markdown(result)
        try:
            import markdown as markdown_lib
            body = markdown_lib.markdown(
                markdown_text,
                extensions=["tables", "fenced_code"],
            )
        except ImportError:
            body = f"<pre>{escape(markdown_text)}</pre>"

        title = f"{result.ticker} Equity Research Report"
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.55; max-width: 1050px; margin: 40px auto; padding: 0 24px; color: #1f2937; }}
h1, h2 {{ color: #111827; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0; }}
th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }}
th {{ background: #f3f4f6; }}
blockquote {{ border-left: 4px solid #9ca3af; margin-left: 0; padding-left: 16px; color: #4b5563; }}
code {{ background: #f3f4f6; padding: 2px 5px; }}
</style>
</head>
<body>{body}</body>
</html>"""

    def save(self, result: Any, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        base = output_dir / f"{result.ticker}_equity_research"
        markdown_path = base.with_suffix(".md")
        html_path = base.with_suffix(".html")
        markdown_path.write_text(self.markdown(result), encoding="utf-8")
        html_path.write_text(self.html(result), encoding="utf-8")
        return {"markdown": markdown_path, "html": html_path}
