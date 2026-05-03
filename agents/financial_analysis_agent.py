"""
agents/financial_analysis_agent.py
------------------------------------
Financial Analysis Agent — evaluates a company's fundamental health.

Responsibilities
----------------
• Analyse the company info returned by the Data Collection Agent.
• Produce a financial health score (1–10).
• Return a human-readable explanation of the reasoning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from utils.helpers import safe_get, format_currency, format_percentage, format_large_number
from utils.scoring import compute_financial_score

logger = logging.getLogger(__name__)


class FinancialAnalysisAgent:
    """
    Analyses fundamental financial data and produces a health score.

    Usage
    -----
    agent  = FinancialAnalysisAgent()
    result = agent.run(data_collection_result)
    """

    def __init__(self) -> None:
        self.name = "Financial Analysis Agent"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse financial fundamentals.

        Parameters
        ----------
        data : dict returned by DataCollectionAgent.run()

        Returns
        -------
        dict with keys:
            success          : bool
            error            : str | None
            financial_score  : float  (1–10)
            score_label      : str    ('Weak' | 'Moderate' | 'Strong' | 'Very Strong')
            summary          : str    one-sentence summary
            reasons          : list[str]  detailed reasoning lines
            metrics          : dict   key financial metrics for display
        """
        result: Dict[str, Any] = {
            "success": False,
            "error": None,
            "financial_score": 5.0,
            "score_label": "Moderate",
            "summary": "",
            "reasons": [],
            "metrics": {},
        }

        if not data.get("success"):
            result["error"] = "Data collection failed — cannot run financial analysis."
            return result

        info: dict = data.get("info", {})
        ticker: str = data.get("ticker", "N/A")
        company: str = data.get("company_name", ticker)

        logger.info("[%s] Analysing fundamentals for %s", self.name, ticker)

        # ── Score ──────────────────────────────────────────────────────────
        financial_score, reasons = compute_financial_score(info)

        # ── Metrics dictionary (for display) ──────────────────────────────
        pe = safe_get(info, "trailingPE") or safe_get(info, "forwardPE")
        forward_pe = safe_get(info, "forwardPE")
        market_cap = data.get("market_cap")
        profit_margin = safe_get(info, "profitMargins")
        revenue_growth = safe_get(info, "revenueGrowth")
        earnings_growth = safe_get(info, "earningsGrowth")
        debt_to_equity = safe_get(info, "debtToEquity")
        current_ratio = safe_get(info, "currentRatio")
        roe = safe_get(info, "returnOnEquity")
        roa = safe_get(info, "returnOnAssets")
        revenue = safe_get(info, "totalRevenue")
        gross_profit = safe_get(info, "grossProfits")
        ebitda = safe_get(info, "ebitda")
        free_cash_flow = safe_get(info, "freeCashflow")
        beta = safe_get(info, "beta")
        dividend_yield = safe_get(info, "dividendYield")
        peg_ratio = safe_get(info, "pegRatio")

        metrics = {
            "Market Cap": format_large_number(market_cap),
            "P/E Ratio (Trailing)": f"{pe:.2f}" if pe else "N/A",
            "P/E Ratio (Forward)": f"{forward_pe:.2f}" if forward_pe else "N/A",
            "PEG Ratio": f"{peg_ratio:.2f}" if peg_ratio else "N/A",
            "Profit Margin": format_percentage(profit_margin),
            "Revenue Growth (YoY)": format_percentage(revenue_growth),
            "Earnings Growth (YoY)": format_percentage(earnings_growth),
            "Return on Equity (ROE)": format_percentage(roe),
            "Return on Assets (ROA)": format_percentage(roa),
            "Debt-to-Equity": f"{debt_to_equity:.2f}" if debt_to_equity else "N/A",
            "Current Ratio": f"{current_ratio:.2f}" if current_ratio else "N/A",
            "Total Revenue": format_large_number(revenue),
            "Gross Profit": format_large_number(gross_profit),
            "EBITDA": format_large_number(ebitda),
            "Free Cash Flow": format_large_number(free_cash_flow),
            "Beta": f"{beta:.2f}" if beta else "N/A",
            "Dividend Yield": format_percentage(dividend_yield),
        }

        # ── Label ─────────────────────────────────────────────────────────
        score_label = self._score_to_label(financial_score)

        # ── Summary sentence ──────────────────────────────────────────────
        summary = (
            f"{company} has a financial health score of {financial_score}/10, "
            f"indicating {score_label.lower()} fundamentals."
        )

        result.update(
            {
                "success": True,
                "financial_score": financial_score,
                "score_label": score_label,
                "summary": summary,
                "reasons": reasons,
                "metrics": metrics,
            }
        )

        logger.info(
            "[%s] %s — score: %.1f (%s)",
            self.name, ticker, financial_score, score_label,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 8.0:
            return "Very Strong"
        if score >= 6.0:
            return "Strong"
        if score >= 4.0:
            return "Moderate"
        return "Weak"
