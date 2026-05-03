"""
utils/indicators.py
-------------------
Pure-Python / NumPy / Pandas implementations of common technical indicators
used by the Technical Analysis Agent.

All functions accept a pandas Series of closing prices and return either
a Series or a scalar float.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def calculate_moving_averages(prices: pd.Series) -> Dict[str, Optional[float]]:
    """
    Calculate the 20-day, 50-day, and 200-day Simple Moving Averages (SMA).

    Parameters
    ----------
    prices : pd.Series
        Time-ordered closing prices (oldest → newest).

    Returns
    -------
    dict with keys 'sma_20', 'sma_50', 'sma_200'.
    Each value is the latest SMA or None if there is insufficient data.
    """
    result: Dict[str, Optional[float]] = {
        "sma_20": None,
        "sma_50": None,
        "sma_200": None,
    }

    if prices is None or len(prices) == 0:
        return result

    clean = prices.dropna()

    if len(clean) >= 20:
        result["sma_20"] = float(clean.rolling(window=20).mean().iloc[-1])

    if len(clean) >= 50:
        result["sma_50"] = float(clean.rolling(window=50).mean().iloc[-1])

    if len(clean) >= 200:
        result["sma_200"] = float(clean.rolling(window=200).mean().iloc[-1])

    return result


def calculate_ema(prices: pd.Series, period: int = 20) -> Optional[float]:
    """
    Calculate the Exponential Moving Average (EMA) for the given period.

    Returns the most recent EMA value, or None if insufficient data.
    """
    clean = prices.dropna()
    if len(clean) < period:
        return None
    ema_series = clean.ewm(span=period, adjust=False).mean()
    return float(ema_series.iloc[-1])


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def calculate_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """
    Calculate the Relative Strength Index (RSI) using the Wilder smoothing method.

    Parameters
    ----------
    prices : pd.Series  Closing prices (oldest → newest).
    period : int        Look-back period (default 14).

    Returns
    -------
    RSI value as a float in [0, 100], or None if insufficient data.

    Interpretation
    --------------
    > 70 → Overbought (potential sell signal)
    < 30 → Oversold  (potential buy signal)
    30–70 → Neutral
    """
    clean = prices.dropna()
    if len(clean) < period + 1:
        return None

    delta = clean.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Initial averages
    avg_gain = gain.iloc[1 : period + 1].mean()
    avg_loss = loss.iloc[1 : period + 1].mean()

    # Wilder smoothing over the remaining data
    for i in range(period + 1, len(clean)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(float(rsi), 2)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def calculate_volatility(prices: pd.Series, window: int = 20) -> Optional[float]:
    """
    Calculate annualised historical volatility as the standard deviation of
    log daily returns, scaled to annual (×√252).

    Parameters
    ----------
    prices : pd.Series  Daily closing prices.
    window : int        Rolling window for std calculation.

    Returns
    -------
    Annualised volatility as a decimal (e.g. 0.25 = 25 %), or None.
    """
    clean = prices.dropna()
    if len(clean) < window + 1:
        return None

    log_returns = np.log(clean / clean.shift(1)).dropna()
    vol = float(log_returns.rolling(window=window).std().iloc[-1]) * np.sqrt(252)
    return round(vol, 4)


# ---------------------------------------------------------------------------
# Recent trend & drawdown
# ---------------------------------------------------------------------------

def calculate_recent_trend(prices: pd.Series, days: int = 20) -> Optional[float]:
    """
    Return the percentage price change over the last `days` trading sessions.

    Positive → uptrend; Negative → downtrend.
    """
    clean = prices.dropna()
    if len(clean) < days + 1:
        days = len(clean) - 1
    if days < 1:
        return None

    start = float(clean.iloc[-(days + 1)])
    end = float(clean.iloc[-1])
    if start == 0:
        return None
    return round((end - start) / start * 100, 2)


def calculate_max_drawdown(prices: pd.Series) -> Optional[float]:
    """
    Calculate the maximum drawdown from peak for the entire price series.

    Returns a negative decimal (e.g. -0.35 = -35 %).
    """
    clean = prices.dropna()
    if len(clean) < 2:
        return None

    cumulative = (1 + clean.pct_change().fillna(0)).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return round(float(drawdown.min()), 4)


# ---------------------------------------------------------------------------
# MACD (bonus indicator)
# ---------------------------------------------------------------------------

def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate MACD line, Signal line, and Histogram.

    Returns (macd_line, signal_line, histogram) — all may be None if
    insufficient data.
    """
    clean = prices.dropna()
    if len(clean) < slow + signal:
        return None, None, None

    ema_fast = clean.ewm(span=fast, adjust=False).mean()
    ema_slow = clean.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return (
        round(float(macd_line.iloc[-1]), 4),
        round(float(signal_line.iloc[-1]), 4),
        round(float(histogram.iloc[-1]), 4),
    )
