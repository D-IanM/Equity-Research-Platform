"""Google Colab-friendly interactive dashboard built with ipywidgets."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .pipeline import EquityResearchPipeline
from .valuation import DCFInputs
from .visualization import (
    dcf_projection_chart,
    drawdown_chart,
    price_dashboard,
    ratio_chart,
    sensitivity_heatmap,
)


def launch_colab_dashboard(project_root: str | Path = ".") -> None:
    """Render an interactive one-company research dashboard."""
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display

    ticker = widgets.Text(value="AAPL", description="Ticker:")
    peers = widgets.Text(
        value="MSFT,GOOGL,AMZN,META",
        description="Peers:",
        layout=widgets.Layout(width="520px"),
    )
    growth = widgets.FloatSlider(
        value=0.08,
        min=-0.05,
        max=0.25,
        step=0.005,
        description="FCF growth:",
        readout_format=".1%",
        style={"description_width": "100px"},
    )
    discount = widgets.FloatSlider(
        value=0.09,
        min=0.06,
        max=0.16,
        step=0.005,
        description="Discount rate:",
        readout_format=".1%",
        style={"description_width": "100px"},
    )
    terminal = widgets.FloatSlider(
        value=0.025,
        min=0.0,
        max=0.05,
        step=0.005,
        description="Terminal g:",
        readout_format=".1%",
        style={"description_width": "100px"},
    )
    include_sec = widgets.Checkbox(
        value=bool(os.getenv("SEC_USER_AGENT")),
        description="Include SEC",
    )
    include_macro = widgets.Checkbox(
        value=bool(os.getenv("FRED_API_KEY")),
        description="Include FRED",
    )
    run_button = widgets.Button(
        description="Run Research",
        button_style="primary",
        icon="play",
    )
    output = widgets.Output()

    controls = widgets.VBox(
        [
            widgets.HBox([ticker, peers]),
            widgets.HBox([growth, discount, terminal]),
            widgets.HBox([include_sec, include_macro, run_button]),
        ]
    )

    def on_run(_: object) -> None:
        with output:
            clear_output(wait=True)
            try:
                config = ProjectConfig(
                    project_root=Path(project_root).resolve()
                )
                pipeline = EquityResearchPipeline(config)
                assumptions = DCFInputs(
                    initial_growth=growth.value,
                    discount_rate=discount.value,
                    terminal_growth=terminal.value,
                )
                peer_list = [
                    p.strip().upper()
                    for p in peers.value.split(",")
                    if p.strip()
                ]
                print("Running the pipeline. External data calls may take a minute...")
                result = pipeline.run(
                    ticker.value,
                    peers=peer_list,
                    dcf_inputs=assumptions,
                    include_sec=include_sec.value,
                    include_macro=include_macro.value,
                    persist=True,
                )
                clear_output(wait=True)
                name = result.bundle.info.get("longName", result.ticker)
                display(HTML(f"<h2>{name} ({result.ticker})</h2>"))
                display(
                    pd.DataFrame(
                        {
                            "Metric": [
                                "Quality Score",
                                "Current Price",
                                "DCF Value / Share",
                                "DCF Upside / Downside",
                                "Sharpe Ratio",
                                "Maximum Drawdown",
                            ],
                            "Value": [
                                f"{result.fundamentals.quality_score:.1f}/10",
                                f"${result.fundamentals.values.get('price', float('nan')):,.2f}",
                                (
                                    f"${result.dcf.intrinsic_value_per_share:,.2f}"
                                    if result.dcf else "N/A"
                                ),
                                (
                                    f"{result.dcf.upside_downside:.1%}"
                                    if result.dcf else "N/A"
                                ),
                                f"{result.risk_metrics.get('sharpe_ratio', float('nan')):.2f}",
                                f"{result.risk_metrics.get('max_drawdown', float('nan')):.1%}",
                            ],
                        }
                    )
                )
                price_dashboard(
                    result.features.tail(1250), result.ticker
                ).show()
                drawdown_chart(
                    pipeline.risk_analyzer.drawdown_series(
                        result.bundle.prices[
                            "Adj Close"
                            if "Adj Close" in result.bundle.prices
                            else "Close"
                        ]
                    ),
                    result.ticker,
                ).show()
                ratio_chart(result.fundamentals.ratios).show()
                if result.dcf:
                    dcf_projection_chart(result.dcf).show()
                    sensitivity_heatmap(
                        result.dcf_sensitivity, result.ticker
                    ).show()
                if not result.comparable_values.empty:
                    display(HTML("<h3>Comparable Valuation</h3>"))
                    display(result.comparable_values)
                display(
                    HTML(
                        f"<p>Reports saved to <code>{config.reports_dir}</code>.</p>"
                    )
                )
            except Exception as exc:
                clear_output(wait=True)
                display(
                    HTML(
                        f"<h3>Pipeline error</h3><pre>{type(exc).__name__}: {exc}</pre>"
                    )
                )

    run_button.on_click(on_run)
    display(controls, output)
