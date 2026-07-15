"""Centralized project configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(slots=True)
class ProjectConfig:
    """Runtime settings shared across notebooks and package modules."""

    project_root: Path = field(
        default_factory=lambda: Path(os.getenv("EQUITY_RESEARCH_ROOT", ".")).resolve()
    )
    start_date: str = "2016-01-01"
    end_date: str | None = None
    benchmark: str = "SPY"
    risk_free_rate: float = 0.04
    trading_days: int = 252
    request_timeout: int = 30
    request_retries: int = 3
    sec_user_agent: str | None = field(
        default_factory=lambda: os.getenv("SEC_USER_AGENT")
    )
    fred_api_key: str | None = field(
        default_factory=lambda: os.getenv("FRED_API_KEY")
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.project_root / "figures"

    def ensure_directories(self) -> None:
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.features_dir,
            self.reports_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
