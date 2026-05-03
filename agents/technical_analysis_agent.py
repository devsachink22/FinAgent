"""
agents/technical_analysis_agent.py
------------------------------------
Technical Analysis Agent — analyses price action and momentum indicators.

Responsibilities
----------------
• Calculate moving averages (20, 50, 200 day).
• Calculate RSI (14-day).
• Calculate annualised volatility.
• Calculate recent price trend and MACD.
• Produce a technical signal: Bullish | Neutral | Bearish.
• Explain the signal in plain language.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.indicators import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_volatility,
    calculate_recent_trend,
    calculate_max_drawdown,
    calculate_macd,
)
from utils.helpers import safe_get

logger = logging.getLogger(__name__)


class TechnicalAnalysisAgent:
    """
    Analyses historical price data using technical indicators.

    Usage
    -----
    agent  = TechnicalAnalysisAgent()
    result = agent.run(data_collection_result)
    """

    def __init__(self) -> None:
        self.name = "Technical Analysis Agent"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform technical analysis.

        Parameters
        ----------
        data : dict returned by DataCollectionAgent.run()

        Returns
        -------
        dict with keys:
            success         : bool
            error           : str | None
            signal          : str  ('Bullish' | 'Neutral' | 'Bearish')
            signal_strength : float (0–1, e.g. 0.7 = moderately bullish)
            summary         : str
            reasons         : list[str]
            indicators      : dict  (all computed indicator values)
            price_history   : pd.Series | None  (closing prices for charts)
            sma_20          : float | None
            sma_50          : float | None
            sma_200         : float | None
            rsi             : float | None
            volatility      : float | None
            recent_trend_pct: float | None
        """
        result: Dict[str, Any] = {
            "success": False,
            "error": None,
            "signal": "Neutral",
            "signal_strength": 0.5,
            "summary": "",
            "reasons": [],
            "indicators": {},
            "price_history": None,
            "sma_20": None,
            "sma_50": None,
            "sma_200": None,
            "rsi": None,
            "volatility": None,
            "recent_trend_pct": None,
        }

        if not data.get("success"):
            result["error"] = "Data collection failed — cannot run technical analysis."
            return result

        history: Optional[pd.DataFrame] = data.get("history")
        ticker: str = data.get("ticker", "N/A")
        current_price: Optional[float] = data.get("current_price")

        if history is None or history.empty:
            result["error"] = (
                f"No historical price data available for {ticker}. "
                "Technical analysis cannot be performed."
            )
            return result

        prices: pd.Series = history["Close"].dropna()
        logger.info("[%s] Analysing %d price points for %s", self.name, len(prices), ticker)

        # ── Compute indicators ─────────────────────────────────────────────
        mas = calculate_moving_averages(prices)
        sma_20 = mas["sma_20"]
        sma_50 = mas["sma_50"]
        sma_200 = mas["sma_200"]

        rsi = calculate_rsi(prices)
        volatility = calculate_volatility(prices)
        recent_trend = calculate_recent_trend(prices, days=20)
        max_dd = calculate_max_drawdown(prices)
        macd_line, signal_line, macd_hist = calculate_macd(prices)

        # ── Scoring ────────────────────────────────────────────────────────
        bullish_points = 0
        bearish_points = 0
        reasons: List[str] = []

        price = current_price or float(prices.iloc[-1])

        # Price vs Moving Averages
        if sma_20 is not None:
            if price > sma_20:
                bullish_points += 1
                reasons.append(
                    f"✅ Price (${price:.2f}) is above the 20-day SMA (${sma_20:.2f}) — short-term uptrend."
                )
            else:
                bearish_points += 1
                reasons.append(
                    f"❌ Price (${price:.2f}) is below the 20-day SMA (${sma_20:.2f}) — short-term weakness."
                )

        if sma_50 is not None:
            if price > sma_50:
                bullish_points += 1
                reasons.append(
                    f"✅ Price is above the 50-day SMA (${sma_50:.2f}) — medium-term uptrend."
                )
            else:
                bearish_points += 1
                reasons.append(
                    f"❌ Price is below the 50-day SMA (${sma_50:.2f}) — medium-term weakness."
                )

        if sma_200 is not None:
            if price > sma_200:
                bullish_points += 1
                reasons.append(
                    f"✅ Price is above the 200-day SMA (${sma_200:.2f}) — long-term bullish structure."
                )
            else:
                bearish_points += 1
                reasons.append(
                    f"❌ Price is below the 200-day SMA (${sma_200:.2f}) — long-term bearish structure."
                )

        # Golden / Death Cross
        if sma_20 is not None and sma_50 is not None:
            if sma_20 > sma_50:
                bullish_points += 1
                reasons.append("✅ 20-day SMA is above 50-day SMA (potential Golden Cross pattern).")
            else:
                bearish_points += 1
                reasons.append("❌ 20-day SMA is below 50-day SMA (potential Death Cross pattern).")

        # RSI
        if rsi is not None:
            reasons.append(f"📊 RSI (14-day): {rsi:.1f}")
            if rsi > 70:
                bearish_points += 1
                reasons.append(
                    f"⚠️ RSI of {rsi:.1f} is in overbought territory (>70). "
                    "Price may be due for a pullback."
                )
            elif rsi < 30:
                bullish_points += 1
                reasons.append(
                    f"✅ RSI of {rsi:.1f} is in oversold territory (<30). "
                    "Price may bounce or reverse upward."
                )
            else:
                reasons.append(f"RSI of {rsi:.1f} is in neutral range (30–70) — no overbought/oversold signal.")

        # Recent trend
        if recent_trend is not None:
            if recent_trend > 5:
                bullish_points += 1
                reasons.append(f"✅ Strong positive price momentum: +{recent_trend:.1f}% over the last 20 days.")
            elif recent_trend < -5:
                bearish_points += 1
                reasons.append(f"❌ Negative price momentum: {recent_trend:.1f}% over the last 20 days.")
            else:
                reasons.append(f"ℹ️ Price is relatively flat ({recent_trend:+.1f}% over 20 days).")

        # MACD
        if macd_line is not None and signal_line is not None:
            if macd_line > signal_line:
                bullish_points += 1
                reasons.append(f"✅ MACD ({macd_line:.3f}) is above the Signal line ({signal_line:.3f}) — bullish momentum.")
            else:
                bearish_points += 1
                reasons.append(f"❌ MACD ({macd_line:.3f}) is below the Signal line ({signal_line:.3f}) — bearish momentum.")

        # Max Drawdown info (informational)
        if max_dd is not None:
            reasons.append(
                f"📉 Maximum historical drawdown: {max_dd*100:.1f}% from peak."
            )

        # ── Determine signal ───────────────────────────────────────────────
        total = bullish_points + bearish_points
        if total == 0:
            signal = "Neutral"
            strength = 0.5
        else:
            bull_ratio = bullish_points / total
            if bull_ratio >= 0.65:
                signal = "Bullish"
                strength = round(bull_ratio, 2)
            elif bull_ratio <= 0.35:
                signal = "Bearish"
                strength = round(1 - bull_ratio, 2)
            else:
                signal = "Neutral"
                strength = 0.5

        # ── Summary sentence ──────────────────────────────────────────────
        summary = (
            f"Technical signal is **{signal}** "
            f"({bullish_points} bullish vs {bearish_points} bearish indicator{'s' if bearish_points != 1 else ''}). "
        )
        if volatility is not None:
            summary += f"Annualised volatility: {volatility*100:.1f}%."

        result.update(
            {
                "success": True,
                "signal": signal,
                "signal_strength": strength,
                "summary": summary,
                "reasons": reasons,
                "price_history": prices,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "rsi": rsi,
                "volatility": volatility,
                "recent_trend_pct": recent_trend,
                "indicators": {
                    "SMA 20": f"${sma_20:.2f}" if sma_20 else "N/A",
                    "SMA 50": f"${sma_50:.2f}" if sma_50 else "N/A",
                    "SMA 200": f"${sma_200:.2f}" if sma_200 else "N/A",
                    "RSI (14)": f"{rsi:.1f}" if rsi else "N/A",
                    "Volatility (Ann.)": f"{volatility*100:.1f}%" if volatility else "N/A",
                    "20-day Trend": f"{recent_trend:+.1f}%" if recent_trend else "N/A",
                    "Max Drawdown": f"{max_dd*100:.1f}%" if max_dd else "N/A",
                    "MACD Line": f"{macd_line:.4f}" if macd_line else "N/A",
                    "MACD Signal": f"{signal_line:.4f}" if signal_line else "N/A",
                },
            }
        )

        logger.info("[%s] %s — signal: %s", self.name, ticker, signal)
        return result
