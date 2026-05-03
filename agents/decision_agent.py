"""
agents/decision_agent.py
-------------------------
Decision Agent — synthesises all agent outputs into a final recommendation.

Responsibilities
----------------
• Combine financial score, technical signal, news sentiment, and risk level.
• Apply rule-based decision logic to produce: Buy | Hold | Sell.
• Compute a confidence score (0–100 %).
• Explain the final decision clearly.
"""

from __future__ import annotations

import logging
import os
import requests
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class DecisionAgent:
    """
    Generates a final investment recommendation.

    Usage
    -----
    agent  = DecisionAgent()
    result = agent.run(financial, technical, sentiment, risk)
    """

    def __init__(self) -> None:
        self.name = "Decision Agent"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        financial: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any],
        risk: Dict[str, Any],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate the final recommendation.

        Parameters
        ----------
        financial  : dict from FinancialAnalysisAgent
        technical  : dict from TechnicalAnalysisAgent
        sentiment  : dict from NewsSentimentAgent
        risk       : dict from RiskAnalysisAgent
        data       : dict from DataCollectionAgent (for company info)

        Returns
        -------
        dict with keys:
            success          : bool
            error            : str | None
            recommendation   : str   ('Buy' | 'Hold' | 'Sell')
            confidence       : float (0–100)
            summary          : str
            reasons          : list[str]
            signal_breakdown : dict  (each agent's contribution)
        """
        result: Dict[str, Any] = {
            "success": False,
            "error": None,
            "recommendation": "Hold",
            "confidence": 50.0,
            "summary": "",
            "reasons": [],
            "signal_breakdown": {},
        }

        ticker: str = data.get("ticker", "N/A")
        company: str = data.get("company_name", ticker)
        logger.info("[%s] Generating recommendation for %s", self.name, ticker)

        # ── Extract signals ────────────────────────────────────────────────
        fin_score: float      = financial.get("financial_score", 5.0)
        fin_label: str        = financial.get("score_label", "Moderate")
        tech_signal: str      = technical.get("signal", "Neutral")
        tech_strength: float  = technical.get("signal_strength", 0.5)
        news_sent: str        = sentiment.get("sentiment", "Neutral")
        risk_level: str       = risk.get("risk_level", "Medium")
        risk_score: float     = risk.get("risk_score", 5.0)

        # ── Map signals to points ──────────────────────────────────────────
        # Buy points (max +10), Sell points (max +10)
        buy_pts = 0.0
        sell_pts = 0.0
        reasons: List[str] = []

        # ── Financial Health ───────────────────────────────────────────────
        if fin_score >= 7.0:
            buy_pts += 3.0
            reasons.append(f"✅ Financial health is {fin_label} ({fin_score}/10) — supports a Buy signal.")
        elif fin_score >= 5.0:
            buy_pts += 1.5
            reasons.append(f"ℹ️ Financial health is {fin_label} ({fin_score}/10) — neither strongly bullish nor bearish.")
        elif fin_score >= 3.5:
            sell_pts += 1.5
            reasons.append(f"⚠️ Financial health is {fin_label} ({fin_score}/10) — elevated fundamental risk.")
        else:
            sell_pts += 3.0
            reasons.append(f"❌ Financial health is {fin_label} ({fin_score}/10) — weak fundamentals suggest caution.")

        # ── Technical Signal ───────────────────────────────────────────────
        if tech_signal == "Bullish":
            buy_pts += 2.5 * tech_strength
            reasons.append(f"✅ Technical signal is {tech_signal} (strength: {tech_strength:.0%}) — price action is positive.")
        elif tech_signal == "Bearish":
            sell_pts += 2.5 * tech_strength
            reasons.append(f"❌ Technical signal is {tech_signal} (strength: {tech_strength:.0%}) — price action is negative.")
        else:
            buy_pts += 1.0
            sell_pts += 1.0
            reasons.append("ℹ️ Technical signal is Neutral — mixed price action.")

        # ── News Sentiment ─────────────────────────────────────────────────
        if news_sent == "Positive":
            buy_pts += 1.5
            reasons.append("✅ News sentiment is Positive — favourable media coverage.")
        elif news_sent == "Negative":
            sell_pts += 1.5
            reasons.append("❌ News sentiment is Negative — adverse press may weigh on the stock.")
        else:
            reasons.append("ℹ️ News sentiment is Neutral — no strong directional bias from the press.")

        # ── Risk Level ─────────────────────────────────────────────────────
        if risk_level == "High":
            sell_pts += 2.5
            reasons.append(f"❌ Risk level is High ({risk_score}/10) — multiple risk factors elevated.")
        elif risk_level == "Low":
            buy_pts += 2.0
            reasons.append(f"✅ Risk level is Low ({risk_score}/10) — risk-adjusted case for owning the stock.")
        else:
            sell_pts += 0.5
            buy_pts += 0.5
            reasons.append(f"ℹ️ Risk level is Medium ({risk_score}/10) — moderate risk.")

        # ── Decision logic ────────────────────────────────────────────────
        total = buy_pts + sell_pts
        if total == 0:
            recommendation = "Hold"
            confidence = 50.0
        else:
            buy_ratio = buy_pts / total

            if buy_ratio >= 0.62:
                recommendation = "Buy"
                confidence = round(50.0 + (buy_ratio - 0.5) * 120, 1)
            elif buy_ratio <= 0.38:
                recommendation = "Sell"
                confidence = round(50.0 + (0.5 - buy_ratio) * 120, 1)
            else:
                recommendation = "Hold"
                # Confidence for Hold is how close to 50/50
                hold_strength = 1.0 - abs(buy_ratio - 0.5) * 4
                confidence = round(max(50.0, 50.0 + hold_strength * 20), 1)

        # Clamp confidence
        confidence = min(99.0, max(1.0, confidence))

        # ── Signal breakdown ──────────────────────────────────────────────
        signal_breakdown = {
            "Financial Health": {
                "value": f"{fin_score}/10 ({fin_label})",
                "points_buy": round(buy_pts, 2) if fin_score >= 5.0 else 0,
                "direction": "Buy" if fin_score >= 5.0 else "Sell",
            },
            "Technical Signal": {
                "value": tech_signal,
                "direction": tech_signal if tech_signal != "Neutral" else "Hold",
            },
            "News Sentiment": {
                "value": news_sent,
                "direction": (
                    "Buy" if news_sent == "Positive"
                    else "Sell" if news_sent == "Negative"
                    else "Hold"
                ),
            },
            "Risk Level": {
                "value": f"{risk_level} ({risk_score}/10)",
                "direction": (
                    "Sell" if risk_level == "High"
                    else "Buy" if risk_level == "Low"
                    else "Hold"
                ),
            },
        }

        # ── Final summary ─────────────────────────────────────────────────
        emoji = {"Buy": "🟢", "Hold": "🟡", "Sell": "🔴"}.get(recommendation, "⚪")
        summary = (
            f"Based on the analysis of {company}, the final recommendation is "
            f"**{emoji} {recommendation}** with {confidence:.0f}% confidence. "
        )
        if recommendation == "Buy":
            summary += (
                "The combination of strong fundamentals, positive technical momentum, "
                "and manageable risk supports initiating or adding to a position."
            )
        elif recommendation == "Sell":
            summary += (
                "Weak signals across multiple dimensions suggest reducing or exiting "
                "the position to manage downside risk."
            )
        else:
            summary += (
                "Mixed signals across agents suggest holding current positions and "
                "monitoring the stock for clearer directional signals."
            )

        reasons.insert(
            0,
            f"🏁 Decision Score — Buy: {buy_pts:.1f} pts vs Sell: {sell_pts:.1f} pts "
            f"(buy ratio: {buy_pts/total*100:.0f}%)" if total > 0 else "🏁 Insufficient data to score.",
        )

        result.update(
            {
                "success": True,
                "recommendation": recommendation,
                "confidence": confidence,
                "summary": summary,
                "reasons": reasons,
                "signal_breakdown": signal_breakdown,
            }
        )

        # ── LLM Summary ───────────────────────────────────────────────────
        if self.openai_api_key:
            logger.info("[%s] Fetching LLM summary from OpenAI...", self.name)
            try:
                prompt = (
                    f"You are a professional financial AI analyst. Write a concise, one-paragraph "
                    f"executive summary (max 3-4 sentences) justifying the decision to {recommendation} "
                    f"stock {ticker} ({company}). Here are the signals:\n"
                    f"- Financial Health: {fin_score}/10 ({fin_label})\n"
                    f"- Technical Signal: {tech_signal}\n"
                    f"- News Sentiment: {news_sent}\n"
                    f"- Risk Level: {risk_level} ({risk_score}/10)\n"
                    f"Focus on the rationale without using emojis."
                )
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.3,
                }
                resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
                if resp.status_code == 200:
                    llm_text = resp.json()["choices"][0]["message"]["content"].strip()
                    result["llm_summary"] = llm_text
                else:
                    logger.warning("[%s] OpenAI API error: %s", self.name, resp.text)
            except Exception as exc:
                logger.error("[%s] Error fetching OpenAI summary: %s", self.name, exc)

        logger.info(
            "[%s] %s → %s (confidence: %.1f%%)",
            self.name, ticker, recommendation, confidence,
        )
        return result
