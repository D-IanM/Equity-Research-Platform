"""Command-line entry point for local or Colab terminal use."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ProjectConfig
from .pipeline import EquityResearchPipeline
from .valuation import DCFInputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end equity research report."
    )
    parser.add_argument("ticker", help="Target ticker, for example AAPL")
    parser.add_argument(
        "--peers",
        nargs="*",
        default=[],
        help="Optional peer tickers.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Root directory for data and report outputs.",
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--growth", type=float, default=0.08)
    parser.add_argument("--discount-rate", type=float, default=0.09)
    parser.add_argument("--terminal-growth", type=float, default=0.025)
    parser.add_argument("--skip-sec", action="store_true")
    parser.add_argument("--skip-fred", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ProjectConfig(
        project_root=Path(args.project_root).resolve(),
        start_date=args.start,
        benchmark=args.benchmark,
    )
    pipeline = EquityResearchPipeline(config)
    result = pipeline.run(
        args.ticker,
        peers=args.peers,
        dcf_inputs=DCFInputs(
            initial_growth=args.growth,
            discount_rate=args.discount_rate,
            terminal_growth=args.terminal_growth,
        ),
        include_sec=not args.skip_sec,
        include_macro=not args.skip_fred,
        persist=True,
    )
    print(
        f"Completed {result.ticker}. "
        f"Reports: {config.reports_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
