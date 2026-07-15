"""Market-data feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import InsufficientDataError


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def add_market_features(
    prices: pd.DataFrame,
    volatility_window: int = 21,
    annualization: int = 252,
) -> pd.DataFrame:
    """Create features without forward-looking data leakage."""
    if "Close" not in prices:
        raise InsufficientDataError("Price history must contain a Close column.")
    frame = prices.copy().sort_index()
    close = pd.to_numeric(frame["Close"], errors="coerce")
    adjusted = (
        pd.to_numeric(frame["Adj Close"], errors="coerce")
        if "Adj Close" in frame
        else close
    )
    frame["Return"] = adjusted.pct_change(fill_method=None)
    frame["LogReturn"] = np.log(adjusted / adjusted.shift(1))
    frame["SMA20"] = adjusted.rolling(20, min_periods=20).mean()
    frame["SMA50"] = adjusted.rolling(50, min_periods=50).mean()
    frame["SMA200"] = adjusted.rolling(200, min_periods=200).mean()
    frame["EMA20"] = adjusted.ewm(span=20, adjust=False).mean()
    frame["Momentum21"] = adjusted.pct_change(21, fill_method=None)
    frame["Momentum63"] = adjusted.pct_change(63, fill_method=None)
    frame["RollingVolatility"] = (
        frame["Return"].rolling(volatility_window, min_periods=volatility_window).std()
        * np.sqrt(annualization)
    )
    frame["RSI14"] = _rsi(adjusted, 14)
    running_peak = adjusted.cummax()
    frame["Drawdown"] = adjusted / running_peak - 1
    if "Volume" in frame:
        volume = pd.to_numeric(frame["Volume"], errors="coerce")
        frame["VolumeSMA20"] = volume.rolling(20, min_periods=20).mean()
        frame["VolumeZScore20"] = (
            volume - volume.rolling(20, min_periods=20).mean()
        ) / volume.rolling(20, min_periods=20).std()
    return frame
