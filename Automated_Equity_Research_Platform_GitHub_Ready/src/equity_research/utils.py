"""Shared validation, retry, serialization, and logging utilities."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import requests

from .exceptions import ValidationError

F = TypeVar("F", bound=Callable[..., Any])


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("equity_research")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


logger = configure_logging()


def validate_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned or not re.fullmatch(r"[A-Z0-9.\-^=]{1,20}", cleaned):
        raise ValidationError(f"Invalid ticker symbol: {ticker!r}")
    return cleaned


def retry(
    attempts: int = 3,
    base_delay: float = 0.75,
    retry_on: tuple[type[BaseException], ...] = (
        requests.RequestException,
        TimeoutError,
        ConnectionError,
    ),
) -> Callable[[F], F]:
    """Retry transient network failures using exponential backoff."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            last_error: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:
                    last_error = exc
                    if attempt == attempts:
                        break
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "%s failed on attempt %s/%s; retrying in %.2fs: %s",
                        func.__name__,
                        attempt,
                        attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            assert last_error is not None
            raise last_error

        return wrapped  # type: ignore[return-value]

    return decorator


def flatten_yfinance_columns(frame: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Normalize the several MultiIndex layouts returned by yfinance."""
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        levels = [list(map(str, level)) for level in df.columns.levels]
        if ticker and ticker in levels[-1]:
            try:
                df = df.xs(ticker, axis=1, level=-1, drop_level=True)
            except KeyError:
                pass
        elif len(df.columns.levels) == 2:
            df.columns = [
                "_".join(str(part) for part in col if str(part) not in ("", "None"))
                for col in df.columns
            ]
    return df


def normalize_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    df = df.loc[~df.index.isna()].sort_index()
    df.index = df.index.tz_convert(None)
    return df[~df.index.duplicated(keep="last")]


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Series,)):
        return {str(k): sanitize_for_json(v) for k, v in value.to_dict().items()}
    if isinstance(value, (pd.DataFrame,)):
        return [
            {str(k): sanitize_for_json(v) for k, v in row.items()}
            for row in value.reset_index().to_dict(orient="records")
        ]
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_for_json(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def save_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(path)
            return path
        except (ImportError, ValueError):
            path = path.with_suffix(".csv")
    frame.to_csv(path)
    return path


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def latest_non_null(series: pd.Series) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    return safe_float(cleaned.iloc[0]) if not cleaned.empty else np.nan


def previous_non_null(series: pd.Series) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    return safe_float(cleaned.iloc[1]) if len(cleaned) > 1 else np.nan


def pct_change_safe(current: float, previous: float) -> float:
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0:
        return np.nan
    return current / previous - 1.0


def divide_safe(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator
