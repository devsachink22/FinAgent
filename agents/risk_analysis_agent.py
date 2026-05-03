"""
agents/risk_analysis_agent.py
-------------------------------
Risk Analysis Agent — aggregates risk signals from all data sources.

Responsibilities
----------------
• Combine volatility, valuation, debt, sentiment, and recent price action.
• Produce a risk level: Low | Medium | High.
• Provide clear, plain-language explanations for the risk assessment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from utils.scoring import compute_risk_score
from utils.helpers import safe_get

logger = logging.getLogger(__name__)


class RiskAnalysisAgent:
    """
    Evaluates investment risk across multiple dimensions.

    Usage
    -----
    agent  = RiskAnalysisAgent()
    result = agent.run(data, technical_result, sentiment_result, financial_result)
    """

    def __init__(self) -> None:
        self.name = "Risk Analysis Agent"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        data: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any],
        financial: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform risk analysis.

        Parameters
        ----------
        data       : dict from DataCollectionAgent
        technical  : dict from TechnicalAnalysisAgent
        sentiment  : dict from NewsSentimentAgent
        financial  : dict from FinancialAnalysisAgent

        Returns
        -------
        dict with keys:
            success     : bool
            error       : str | None
            risk_score  : float (1–10)
            risk_level  : str   ('Low' | 'Medium' | 'High')
            summary     : str
            reasons     : list[str]
            risk_factors: dict  (labelled factors for display)
        """
        result: Dict[str, Any] = {
            "success": False,
            "error": None,
            "risk_score": 5.0,
            "risk_level": "Medium",
            "summary": "",
            "reasons": [],
            "risk_factors": {},
        }

        if not data.get("success"):
            result["error"] = "Data collection failed — cannot run risk analysis."
            return result

        ticker: str = data.get("ticker", "N/A")
        company: str = data.get("company_name", ticker)
        info: dict = data.get("info", {})

        logger.info("[%s] Assessing risk for %s", self.name, ticker)

        # ── Gather inputs ──────────────────────────────────────────────────
        volatility: Optional[float] = technical.get("volatility")
        pe_ratio: Optional[float] = data.get("pe_ratio")
        debt_to_equity: Optional[float] = data.get("debt_to_equity")
        news_sentiment: str = sentiment.get("sentiment", "Neutral")
        recent_trend: Optional[float] = technical.get("recent_trend_pct")
        fin_score: float = financial.get("financial_score", 5.0)

        # ── Compute risk score ─────────────────────────────────────────────
        risk_score, risk_level, reasons = compute_risk_score(
            volatility=volatility,
            pe_ratio=pe_ratio,
            debt_to_equity=debt_to_equity,
            sentiment=news_sentiment,
            recent_trend_pct=recent_trend,
            financial_score=fin_score,
        )

        # ── Additional contextual risk factors ────────────────────────────
        beta = safe_get(info, "beta")
        if beta is not None:
            if beta > 1.5:
                reasons.append(
                    f"⚠️ High beta ({beta:.2f}) — stock moves more than the market. "
                    "Amplified gains AND losses."
                )
            elif beta > 1.0:
                reasons.append(f"ℹ️ Beta ({beta:.2f}) slightly above 1 — moderately sensitive to market moves.")
            elif beta > 0:
                reasons.append(f"✅ Low beta ({beta:.2f}) — less sensitive to market swings.")
            else:
                reasons.append(f"ℹ️ Negative beta ({beta:.2f}) — stock may move inversely to the market.")

        # Short interest / insider holding (informational if available)
        short_pct = safe_get(info, "shortPercentOfFloat")
        if short_pct is not None:
            if short_pct > 0.15:
                reasons.append(
                    f"⚠️ High short interest ({short_pct*100:.1f}% of float shorted) — "
                    "bearish sentiment among traders, but also short-squeeze potential."
                )
            else:
                reasons.append(f"ℹ️ Short interest is {short_pct*100:.1f}% of float — not alarming.")

        # 52-week range
        week52_high = safe_get(info, "fiftyTwoWeekHigh")
        week52_low = safe_get(info, "fiftyTwoWeekLow")
        current_price = data.get("current_price")
        if week52_high and week52_low and current_price:
            range_pct = (week52_high - week52_low) / week52_low * 100
            from_high = (current_price - week52_high) / week52_high * 100
            reasons.append(
                f"📊 52-week range: ${week52_low:.2f} – ${week52_high:.2f} "
                f"(range width: {range_pct:.1f}%). "
                f"Current price is {from_high:.1f}% from 52-week high."
            )

        # ── Build risk factors dict for display ────────────────────────────
        risk_factors = {
            "Annualised Volatility": f"{volatility*100:.1f}%" if volatility else "N/A",
            "Beta": f"{beta:.2f}" if beta else "N/A",
            "P/E Ratio": f"{pe_ratio:.1f}" if pe_ratio else "N/A",
            "Debt-to-Equity": f"{debt_to_equity:.1f}%" if debt_to_equity else "N/A",
            "News Sentiment": news_sentiment,
            "20-day Price Trend": f"{recent_trend:+.1f}%" if recent_trend else "N/A",
            "Financial Health Score": f"{fin_score:.1f}/10",
            "Short Interest": f"{short_pct*100:.1f}%" if short_pct else "N/A",
        }

        # ── Summary ────────────────────────────────────────────────────────
        emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level, "⚪")
        summary = (
            f"{company} has a {emoji} **{risk_level} risk** profile "
            f"(score: {risk_score}/10). "
        )
        if risk_level == "High":
            summary += "Proceed with caution — multiple risk factors are elevated."
        elif risk_level == "Low":
            summary += "Risk appears manageable across key dimensions."
        else:
            summary += "Risk is at a moderate level — balanced outlook."

        result.update(
            {
                "success": True,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "summary": summary,
                "reasons": reasons,
                "risk_factors": risk_factors,
            }
        )

        logger.info("[%s] %s — risk: %s (%.1f)", self.name, ticker, risk_level, risk_score)
        return result
