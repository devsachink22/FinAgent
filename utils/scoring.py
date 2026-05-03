"""
utils/scoring.py
----------------
Rule-based scoring functions used by the Financial Analysis Agent and the
Risk Analysis Agent.  All logic is deterministic and requires no paid APIs.

Scoring philosophy
------------------
• Financial health score  → 1 (very weak) … 10 (very strong)
• Risk score              → 1 (very low)  … 10 (very high)

The functions return both a numeric score and a list of human-readable
reasoning strings so agents can expose "why" to the user.
"""

from typing import Dict, List, Optional, Tuple
from utils.helpers import safe_get, clamp


# ===========================================================================
# Financial Health Score
# ===========================================================================

def compute_financial_score(
    info: dict,
) -> Tuple[float, List[str]]:
    """
    Compute a financial health score (1–10) from yfinance company `info`.

    Scoring breakdown (max 10 points):
    ─────────────────────────────────
    Profitability   up to +3 pts   (profit margins, ROE)
    Valuation       up to +2 pts   (P/E ratio)
    Growth          up to +2 pts   (revenue growth, earnings growth)
    Debt safety     up to +3 pts   (debt-to-equity, current ratio)

    Parameters
    ----------
    info : dict   yfinance Ticker.info dictionary.

    Returns
    -------
    (score, reasons)
    score   : float in [1, 10]
    reasons : list of human-readable explanation strings
    """
    score = 5.0   # neutral starting point
    reasons: List[str] = []

    # ── Profitability ────────────────────────────────────────────────────────
    profit_margin: Optional[float] = safe_get(info, "profitMargins")
    roe: Optional[float] = safe_get(info, "returnOnEquity")

    if profit_margin is not None:
        if profit_margin > 0.20:
            score += 1.5
            reasons.append(f"✅ Strong profit margin of {profit_margin*100:.1f}% (>20%).")
        elif profit_margin > 0.10:
            score += 0.75
            reasons.append(f"✅ Healthy profit margin of {profit_margin*100:.1f}% (10–20%).")
        elif profit_margin > 0.0:
            reasons.append(f"⚠️ Thin but positive profit margin of {profit_margin*100:.1f}%.")
        else:
            score -= 1.5
            reasons.append(f"❌ Negative profit margin of {profit_margin*100:.1f}% — company is losing money.")
    else:
        reasons.append("ℹ️ Profit margin data not available.")

    if roe is not None:
        if roe > 0.15:
            score += 1.0
            reasons.append(f"✅ Good return on equity (ROE) of {roe*100:.1f}% (>15%).")
        elif roe > 0.0:
            score += 0.25
            reasons.append(f"⚠️ Modest ROE of {roe*100:.1f}%.")
        else:
            score -= 0.75
            reasons.append(f"❌ Negative ROE of {roe*100:.1f}% — equity is being eroded.")

    # ── Valuation ────────────────────────────────────────────────────────────
    pe_ratio: Optional[float] = safe_get(info, "trailingPE")
    forward_pe: Optional[float] = safe_get(info, "forwardPE")
    effective_pe = pe_ratio or forward_pe

    if effective_pe is not None:
        if effective_pe < 0:
            score -= 1.0
            reasons.append(f"❌ Negative P/E ratio ({effective_pe:.1f}) — company is not profitable on a trailing basis.")
        elif effective_pe < 15:
            score += 1.0
            reasons.append(f"✅ Low P/E ratio of {effective_pe:.1f} — potentially undervalued.")
        elif effective_pe < 25:
            score += 0.5
            reasons.append(f"✅ Moderate P/E ratio of {effective_pe:.1f} — fairly valued.")
        elif effective_pe < 50:
            score -= 0.25
            reasons.append(f"⚠️ Elevated P/E ratio of {effective_pe:.1f} — market pricing in high growth.")
        else:
            score -= 1.0
            reasons.append(f"❌ Very high P/E ratio of {effective_pe:.1f} — significant valuation risk.")
    else:
        reasons.append("ℹ️ P/E ratio data not available.")

    # ── Growth ───────────────────────────────────────────────────────────────
    revenue_growth: Optional[float] = safe_get(info, "revenueGrowth")
    earnings_growth: Optional[float] = safe_get(info, "earningsGrowth")

    if revenue_growth is not None:
        if revenue_growth > 0.20:
            score += 1.0
            reasons.append(f"✅ Strong revenue growth of {revenue_growth*100:.1f}% YoY.")
        elif revenue_growth > 0.05:
            score += 0.5
            reasons.append(f"✅ Solid revenue growth of {revenue_growth*100:.1f}% YoY.")
        elif revenue_growth > 0.0:
            reasons.append(f"⚠️ Modest revenue growth of {revenue_growth*100:.1f}% YoY.")
        else:
            score -= 0.75
            reasons.append(f"❌ Declining revenue ({revenue_growth*100:.1f}% YoY).")
    else:
        reasons.append("ℹ️ Revenue growth data not available.")

    if earnings_growth is not None:
        if earnings_growth > 0.15:
            score += 1.0
            reasons.append(f"✅ Strong earnings growth of {earnings_growth*100:.1f}% YoY.")
        elif earnings_growth > 0.0:
            score += 0.25
            reasons.append(f"✅ Positive earnings growth of {earnings_growth*100:.1f}% YoY.")
        else:
            score -= 0.5
            reasons.append(f"❌ Earnings contraction of {earnings_growth*100:.1f}% YoY.")

    # ── Debt Safety ──────────────────────────────────────────────────────────
    debt_to_equity: Optional[float] = safe_get(info, "debtToEquity")
    current_ratio: Optional[float] = safe_get(info, "currentRatio")

    if debt_to_equity is not None:
        if debt_to_equity < 50:
            score += 1.5
            reasons.append(f"✅ Low debt-to-equity of {debt_to_equity:.1f}% — conservatively financed.")
        elif debt_to_equity < 100:
            score += 0.5
            reasons.append(f"✅ Moderate debt-to-equity of {debt_to_equity:.1f}%.")
        elif debt_to_equity < 200:
            score -= 0.5
            reasons.append(f"⚠️ Elevated debt-to-equity of {debt_to_equity:.1f}%.")
        else:
            score -= 1.5
            reasons.append(f"❌ High debt-to-equity of {debt_to_equity:.1f}% — significant leverage risk.")
    else:
        reasons.append("ℹ️ Debt-to-equity data not available.")

    if current_ratio is not None:
        if current_ratio > 2.0:
            score += 0.5
            reasons.append(f"✅ Strong current ratio of {current_ratio:.2f} — good short-term liquidity.")
        elif current_ratio > 1.0:
            reasons.append(f"✅ Adequate current ratio of {current_ratio:.2f}.")
        else:
            score -= 1.0
            reasons.append(f"❌ Current ratio below 1.0 ({current_ratio:.2f}) — potential liquidity concern.")

    # ── Final clamp ──────────────────────────────────────────────────────────
    final_score = round(clamp(score, 1.0, 10.0), 1)
    return final_score, reasons


# ===========================================================================
# Risk Score
# ===========================================================================

def compute_risk_score(
    volatility: Optional[float],
    pe_ratio: Optional[float],
    debt_to_equity: Optional[float],
    sentiment: str,
    recent_trend_pct: Optional[float],
    financial_score: float,
) -> Tuple[float, str, List[str]]:
    """
    Compute a risk score (1–10) and a risk level label.

    Parameters
    ----------
    volatility        : Annualised volatility decimal (e.g. 0.30 = 30 %).
    pe_ratio          : Trailing P/E ratio (or None).
    debt_to_equity    : Debt-to-equity ratio in % (or None).
    sentiment         : 'Positive', 'Neutral', or 'Negative'.
    recent_trend_pct  : % price change over last 20 days (or None).
    financial_score   : Score from compute_financial_score (1–10).

    Returns
    -------
    (risk_score, risk_level, reasons)
    risk_score  : float in [1, 10]
    risk_level  : 'Low' | 'Medium' | 'High'
    reasons     : list of human-readable strings
    """
    risk = 3.0   # baseline: below-medium risk
    reasons: List[str] = []

    # ── Volatility ───────────────────────────────────────────────────────────
    if volatility is not None:
        if volatility > 0.60:
            risk += 3.0
            reasons.append(f"⚠️ Very high annualised volatility ({volatility*100:.1f}%) — extreme price swings expected.")
        elif volatility > 0.40:
            risk += 2.0
            reasons.append(f"⚠️ High volatility ({volatility*100:.1f}%) — significant price swings.")
        elif volatility > 0.25:
            risk += 1.0
            reasons.append(f"ℹ️ Moderate volatility ({volatility*100:.1f}%).")
        else:
            risk -= 0.5
            reasons.append(f"✅ Low volatility ({volatility*100:.1f}%) — relatively stable price action.")
    else:
        reasons.append("ℹ️ Volatility data not available.")

    # ── Valuation Risk ───────────────────────────────────────────────────────
    if pe_ratio is not None and pe_ratio > 0:
        if pe_ratio > 60:
            risk += 1.5
            reasons.append(f"⚠️ Extremely high P/E ({pe_ratio:.1f}) — priced for perfection, vulnerable to disappointment.")
        elif pe_ratio > 40:
            risk += 0.75
            reasons.append(f"ℹ️ High P/E ({pe_ratio:.1f}) — elevated valuation risk.")

    # ── Debt Risk ────────────────────────────────────────────────────────────
    if debt_to_equity is not None:
        if debt_to_equity > 200:
            risk += 1.5
            reasons.append(f"⚠️ Very high debt-to-equity ({debt_to_equity:.1f}%) — high financial leverage risk.")
        elif debt_to_equity > 100:
            risk += 0.75
            reasons.append(f"ℹ️ Elevated debt-to-equity ({debt_to_equity:.1f}%).")

    # ── Sentiment Risk ───────────────────────────────────────────────────────
    if sentiment == "Negative":
        risk += 1.0
        reasons.append("⚠️ Negative news sentiment adds downside risk.")
    elif sentiment == "Positive":
        risk -= 0.25
        reasons.append("✅ Positive news sentiment slightly reduces perceived risk.")

    # ── Recent Price Action ──────────────────────────────────────────────────
    if recent_trend_pct is not None:
        if recent_trend_pct < -15:
            risk += 1.5
            reasons.append(f"⚠️ Sharp recent decline ({recent_trend_pct:.1f}% over 20 days) — momentum risk.")
        elif recent_trend_pct < -5:
            risk += 0.75
            reasons.append(f"ℹ️ Moderate recent decline ({recent_trend_pct:.1f}% over 20 days).")
        elif recent_trend_pct > 20:
            risk += 0.5
            reasons.append(f"ℹ️ Rapid recent gain ({recent_trend_pct:.1f}%) — may be overextended, mean-reversion risk.")

    # ── Weak Fundamentals ────────────────────────────────────────────────────
    if financial_score <= 3.5:
        risk += 1.0
        reasons.append("⚠️ Weak financial health amplifies overall risk.")
    elif financial_score >= 7.5:
        risk -= 0.5
        reasons.append("✅ Strong financial fundamentals act as a risk buffer.")

    # ── Map to level ─────────────────────────────────────────────────────────
    risk_score = round(clamp(risk, 1.0, 10.0), 1)

    if risk_score <= 3.5:
        risk_level = "Low"
    elif risk_score <= 6.5:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return risk_score, risk_level, reasons
