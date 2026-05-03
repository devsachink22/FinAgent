"""
app.py
FinAgent Streamlit Dashboard
-----------------------------
Modern finance dashboard for multi-agent stock analysis.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Path setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from agents.data_collection_agent import DataCollectionAgent
from agents.financial_analysis_agent import FinancialAnalysisAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.risk_analysis_agent import RiskAnalysisAgent
from agents.decision_agent import DecisionAgent
from agents.report_agent import ReportAgent
from utils.helpers import is_valid_ticker, format_large_number

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinAgent — Stock Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e8eaf6;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.2rem 1rem;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.45);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Sidebar Text Color */
[data-testid="stSidebar"] .stMarkdown p, 
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown th,
[data-testid="stSidebar"] .stMarkdown td {
    color: #ffffff;
}

/* Text input */
.stTextInput div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
.stTextInput div[data-baseweb="input"] {
    background-color: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}
.stTextInput input {
    background-color: transparent !important;
    color: white !important;
    font-size: 1rem;
    padding: 0.6rem 1rem;
}
.stTextInput input::placeholder {
    color: #ffffff !important;
    opacity: 0.7 !important; /* Slightly transparent white so it still looks like a placeholder */
}
.stTextInput div[data-baseweb="input"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3) !important;
}

/* Select box */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 10px;
    color: white;
}

/* Input Labels */
.stTextInput label, .stSelectbox label, .stTextInput label p, .stSelectbox label p {
    color: #ffffff !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    font-weight: 500;
}

/* Info/Warning/Success/Error */
.stAlert {
    border-radius: 12px;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    border-radius: 99px;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e3a5f 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    border: 1px solid rgba(99, 102, 241, 0.3);
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.hero-title .gradient-text {
    background: linear-gradient(135deg, #a5b4fc, #c4b5fd, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
}

/* Recommendation badge */
.rec-badge {
    display: inline-block;
    padding: 0.4rem 1.5rem;
    border-radius: 99px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}
.rec-buy  { background: linear-gradient(135deg,#064e3b,#065f46); color:#34d399; border:1px solid #34d399; }
.rec-hold { background: linear-gradient(135deg,#713f12,#92400e); color:#fbbf24; border:1px solid #fbbf24; }
.rec-sell { background: linear-gradient(135deg,#7f1d1d,#991b1b); color:#f87171; border:1px solid #f87171; }

/* Section headings */
h2, h3 { color: #a5b4fc; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 FinAgent")
    st.markdown("*Multi-Agent Stock Research*")
    st.markdown("---")

    st.markdown("### 🚀 How to Use")
    st.markdown("""
1. Enter a **stock ticker** (e.g. `AAPL`)
2. Choose an **analysis depth**
3. Click **Analyse Stock**
4. Explore tabs & download the report
""")

    st.markdown("### 📌 Example Tickers")
    st.markdown("""
| Ticker | Company |
|--------|---------|
| `AAPL` | Apple |
| `MSFT` | Microsoft |
| `NVDA` | NVIDIA |
| `TSLA` | Tesla |
| `GOOGL` | Alphabet |
| `AMZN` | Amazon |
| `META` | Meta |
""")

    st.markdown("### 📖 Recommendation Guide")
    st.markdown("""
🟢 **Buy** — Strong fundamentals, bullish signals, manageable risk.

🟡 **Hold** — Mixed signals, wait for clearer direction.

🔴 **Sell** — Weak fundamentals, bearish signals, or high risk.
""")

    st.markdown("### 🛠 Tech Stack")
    st.markdown("""
- `yfinance` — Market data
- `pandas` / `numpy` — Analysis
- `TextBlob` — Sentiment NLP
- `matplotlib` — Charts
- `streamlit` — Dashboard
""")

    st.markdown("---")

# ── Hero Banner ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">📊 <span class="gradient-text">FinAgent</span></div>
    <div class="hero-sub">
        Multi-Agent Stock Research &amp; Recommendation System<br>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input Section ─────────────────────────────────────────────────────────
col_inp1, col_inp2, col_inp3 = st.columns([2, 2, 1])

with col_inp1:
    ticker_input = st.text_input(
        "🔍 Stock Ticker Symbol",
        placeholder="e.g. AAPL, MSFT, TSLA, NVDA",
        max_chars=10,
        help="Enter the official stock ticker symbol (1–10 characters).",
    ).strip().upper()

with col_inp2:
    depth_options = {
        "⚡ Quick Analysis": "quick",
        "📊 Standard Analysis": "standard",
        "🔬 Detailed Analysis": "detailed",
    }
    depth_label = st.selectbox(
        "📐 Analysis Depth",
        list(depth_options.keys()),
        index=1,
        help="Detailed mode shows extra metrics and full reasoning.",
    )
    depth = depth_options[depth_label]

with col_inp3:
    st.markdown("<br>", unsafe_allow_html=True)
    run_analysis = st.button("🚀 Analyse Stock", use_container_width=True)

st.markdown("---")

# ── Pipeline runner ───────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def run_pipeline(ticker: str) -> dict:
    """Run all 7 agents and return results dict. Cached for 5 minutes."""
    results: dict = {}

    data_agent = DataCollectionAgent()
    data = data_agent.run(ticker)
    if not data.get("success"):
        return {"error": data.get("error", "Data collection failed.")}
    results["data"] = data

    fin_agent = FinancialAnalysisAgent()
    results["financial"] = fin_agent.run(data)

    tech_agent = TechnicalAnalysisAgent()
    results["technical"] = tech_agent.run(data)

    news_agent = NewsSentimentAgent()
    results["sentiment"] = news_agent.run(data)

    risk_agent = RiskAnalysisAgent()
    results["risk"] = risk_agent.run(
        data, results["technical"], results["sentiment"], results["financial"]
    )

    dec_agent = DecisionAgent()
    results["decision"] = dec_agent.run(
        results["financial"], results["technical"],
        results["sentiment"], results["risk"], data
    )

    rep_agent = ReportAgent()
    results["report"] = rep_agent.run(
        data, results["financial"], results["technical"],
        results["sentiment"], results["risk"], results["decision"]
    )

    return results

# ── Chart helpers ─────────────────────────────────────────────────────────
def make_price_chart(prices: pd.Series, sma20, sma50, sma200, ticker: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    last_180 = prices.iloc[-180:]
    ax.plot(last_180.index, last_180.values, color="#6366f1", linewidth=1.8,
            label="Price", zorder=3)
    ax.fill_between(last_180.index, last_180.values,
                    alpha=0.08, color="#6366f1")

    if sma20 is not None:
        sma20_series = prices.rolling(20).mean().iloc[-180:]
        ax.plot(last_180.index, sma20_series.values,
                color="#f59e0b", linewidth=1.2, linestyle="--", label="SMA 20", alpha=0.9)
    if sma50 is not None:
        sma50_series = prices.rolling(50).mean().iloc[-180:]
        ax.plot(last_180.index, sma50_series.values,
                color="#10b981", linewidth=1.2, linestyle="--", label="SMA 50", alpha=0.9)
    if sma200 is not None and len(prices) >= 200:
        sma200_series = prices.rolling(200).mean().iloc[-180:]
        ax.plot(last_180.index, sma200_series.values,
                color="#ef4444", linewidth=1.2, linestyle=":", label="SMA 200", alpha=0.9)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.yaxis.label.set_color("#94a3b8")
    ax.grid(axis="y", color="#1e293b", linewidth=0.7)
    ax.set_title(f"{ticker} — Price & Moving Averages (180 days)",
                 color="#a5b4fc", fontsize=11, fontweight="600", pad=10)
    ax.legend(facecolor="#1e293b", edgecolor="#334155",
              labelcolor="#e2e8f0", fontsize=8)
    fig.tight_layout()
    return fig


def make_rsi_chart(prices: pd.Series, ticker: str):
    from utils.indicators import calculate_rsi
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Rolling RSI for chart
    rsi_vals = []
    for i in range(14, len(prices)):
        r = calculate_rsi(prices.iloc[max(0, i-60):i+1])
        rsi_vals.append(r if r is not None else 50.0)

    if not rsi_vals:
        return fig

    rsi_index = prices.index[14:]
    last_n = min(180, len(rsi_vals))
    rsi_plot = rsi_vals[-last_n:]
    idx_plot = rsi_index[-last_n:]

    ax.plot(idx_plot, rsi_plot, color="#8b5cf6", linewidth=1.5, label="RSI (14)")
    ax.axhline(70, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(30, color="#10b981", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.fill_between(idx_plot, rsi_plot, 70,
                    where=[v > 70 for v in rsi_plot],
                    alpha=0.15, color="#ef4444")
    ax.fill_between(idx_plot, rsi_plot, 30,
                    where=[v < 30 for v in rsi_plot],
                    alpha=0.15, color="#10b981")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 30, 50, 70, 100])
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="y", color="#1e293b", linewidth=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title(f"{ticker} — RSI (14-day)", color="#a5b4fc",
                 fontsize=10, fontweight="600", pad=8)
    ax.legend(facecolor="#1e293b", edgecolor="#334155",
              labelcolor="#e2e8f0", fontsize=8)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD  (renders after user clicks Analyse Stock)
# ═══════════════════════════════════════════════════════════════════════════

def render_dashboard(results: dict, depth: str) -> None:
    """Render the full analysis dashboard from pipeline results."""

    data       = results["data"]
    financial  = results["financial"]
    technical  = results["technical"]
    sentiment  = results["sentiment"]
    risk       = results["risk"]
    decision   = results["decision"]
    report     = results["report"]

    ticker   = data.get("ticker", "N/A")
    company  = data.get("company_name", ticker)
    price    = data.get("current_price")
    sector   = data.get("sector", "N/A")
    industry = data.get("industry", "N/A")
    mktcap   = data.get("market_cap")

    rec      = decision.get("recommendation", "Hold")
    conf     = decision.get("confidence", 50.0)
    fin_sc   = financial.get("financial_score", 5.0)
    tech_sig = technical.get("signal", "Neutral")
    news_sen = sentiment.get("sentiment", "Neutral")
    risk_lv  = risk.get("risk_level", "Medium")

    # ── Recommendation header ─────────────────────────────────────────────
    rec_css  = {"Buy": "rec-buy", "Hold": "rec-hold", "Sell": "rec-sell"}.get(rec, "rec-hold")
    rec_icon = {"Buy": "🟢", "Hold": "🟡", "Sell": "🔴"}.get(rec, "⚪")

    st.markdown(f"""
    <div style="text-align:center; padding:1.5rem 1rem 1rem;
                background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(139,92,246,0.10));
                border-radius:16px; border:1px solid rgba(99,102,241,0.25); margin-bottom:1.5rem;">
        <div style="font-size:1rem; color:#94a3b8; margin-bottom:0.3rem;">
            {company} &nbsp;·&nbsp; <code>{ticker}</code> &nbsp;·&nbsp; {sector}
        </div>
        <div style="font-size:2.6rem; font-weight:700; color:#e2e8f0; margin-bottom:0.4rem;">
            {"${:.2f}".format(price) if price else "N/A"}
        </div>
        <span class="rec-badge {rec_css}">{rec_icon} {rec}</span>
        <div style="color:#64748b; font-size:0.85rem; margin-top:0.6rem;">
            Confidence: {conf:.0f}% &nbsp;|&nbsp; Market Cap: {format_large_number(mktcap)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 8 KPI metric cards ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)

    with c1:
        st.metric("💵 Current Price",
                  f"${price:.2f}" if price else "N/A")
    with c2:
        st.metric("🏢 Market Cap", format_large_number(mktcap))
    with c3:
        pe = data.get("pe_ratio")
        st.metric("📊 P/E Ratio", f"{pe:.1f}" if pe else "N/A")
    with c4:
        pm = data.get("profit_margins")
        st.metric("💰 Profit Margin",
                  f"{pm*100:.1f}%" if pm else "N/A")
    with c5:
        st.metric("💪 Financial Score", f"{fin_sc:.1f} / 10",
                  delta=financial.get("score_label",""))
    with c6:
        sig_delta = {"Bullish":"📈 Bullish","Neutral":"➡️ Neutral","Bearish":"📉 Bearish"}.get(tech_sig,"")
        st.metric("📈 Tech Signal", tech_sig)
    with c7:
        sent_icon = {"Positive":"😊","Neutral":"😐","Negative":"😟"}.get(news_sen,"")
        st.metric("📰 Sentiment", f"{sent_icon} {news_sen}")
    with c8:
        risk_icon = {"Low":"🟢","Medium":"🟡","High":"🔴"}.get(risk_lv,"⚪")
        st.metric("⚠️ Risk Level", f"{risk_icon} {risk_lv}",
                  delta=f"{risk.get('risk_score',5):.1f} / 10")

    st.markdown("---")

    # ── TABS ──────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🏠 Overview",
        "💰 Financial",
        "📈 Technical",
        "📰 Sentiment",
        "⚠️ Risk",
        "🏁 Decision",
        "📄 Full Report",
    ])

    # ─────────────────────────────────────────────────────────────────────
    # TAB 0 — OVERVIEW
    # ─────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Company Overview")
        oc1, oc2 = st.columns([1, 2])
        with oc1:
            info = data.get("info", {})
            st.markdown(f"""
**Company:** {company}
**Ticker:** `{ticker}`
**Sector:** {sector}
**Industry:** {industry}
**Exchange:** {info.get('exchange', 'N/A')}
**Employees:** {info.get('fullTimeEmployees', 'N/A'):,} """ if isinstance(info.get('fullTimeEmployees'), int) else f"""
**Company:** {company}
**Ticker:** `{ticker}`
**Sector:** {sector}
**Industry:** {industry}
""")
            st.info(info.get("longBusinessSummary", "No company description available.")[:500] + "…"
                    if info.get("longBusinessSummary") else "No company description available.")

        with oc2:
            prices = technical.get("price_history")
            if prices is not None:
                fig = make_price_chart(
                    prices,
                    technical.get("sma_20"),
                    technical.get("sma_50"),
                    technical.get("sma_200"),
                    ticker,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.warning("Price chart unavailable — no historical data.")

        # Quick summary table
        st.markdown("### Agent Summary")
        summary_data = {
            "Agent": ["💰 Financial", "📈 Technical", "📰 Sentiment", "⚠️ Risk", "🏁 Decision"],
            "Output": [
                f"{fin_sc}/10 ({financial.get('score_label','')})",
                tech_sig,
                news_sen,
                risk_lv,
                f"{rec} ({conf:.0f}% confidence)",
            ],
            "Summary": [
                financial.get("summary",""),
                technical.get("summary",""),
                sentiment.get("summary",""),
                risk.get("summary",""),
                decision.get("summary","")[:120] + "…",
            ],
        }
        st.dataframe(
            summary_data,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        # Confidence progress bar
        st.markdown("### 🏁 Final Confidence Score")
        conf_col1, conf_col2 = st.columns([3,1])
        with conf_col1:
            st.progress(int(conf) / 100)
        with conf_col2:
            st.markdown(f"**{conf:.0f}%** — {rec}")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 1 — FINANCIAL
    # ─────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("💰 Financial Analysis Agent")
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            st.markdown(f"### Score: **{fin_sc:.1f} / 10**")
            score_pct = fin_sc / 10
            st.progress(score_pct)
            label_color = (
                "#34d399" if fin_sc >= 7 else
                "#fbbf24" if fin_sc >= 5 else "#f87171"
            )
            st.markdown(
                f"<div style='color:{label_color};font-size:1.4rem;font-weight:700;'>"
                f"{financial.get('score_label','')}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(financial.get("summary",""))

        with fc2:
            st.markdown("### Key Metrics")
            metrics = financial.get("metrics", {})
            if metrics:
                m_items = list(metrics.items())
                half = len(m_items) // 2
                mc1, mc2 = st.columns(2)
                with mc1:
                    for k, v in m_items[:half]:
                        st.metric(k, v)
                with mc2:
                    for k, v in m_items[half:]:
                        st.metric(k, v)

        st.markdown("---")
        with st.expander("📋 Detailed Financial Reasoning", expanded=(depth == "detailed")):
            for r in financial.get("reasons", []):
                st.markdown(f"- {r}")

        with st.expander("📚 What do these terms mean?"):
            st.markdown("""
| Term | Explanation |
|------|-------------|
| **P/E Ratio** | Price-to-Earnings. Higher = more expensive relative to profits. |
| **Profit Margin** | % of revenue kept as profit after all expenses. |
| **ROE** | Return on Equity — profit generated per dollar of shareholders' equity. |
| **Debt-to-Equity** | How much debt the company uses vs. equity. Lower is generally safer. |
| **Current Ratio** | Short-term assets ÷ short-term liabilities. >1 means good liquidity. |
| **EBITDA** | Earnings before interest, tax, depreciation & amortisation — operating profit proxy. |
| **Free Cash Flow** | Cash left after capital expenditures — the real money available to investors. |
""")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 2 — TECHNICAL
    # ─────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("📈 Technical Analysis Agent")

        sig_color = {"Bullish":"#34d399","Neutral":"#fbbf24","Bearish":"#f87171"}.get(tech_sig,"#94a3b8")
        st.markdown(
            f"<h3 style='color:{sig_color};'>Signal: {tech_sig}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(technical.get("summary",""))

        prices = technical.get("price_history")
        if prices is not None:
            tc1, tc2 = st.columns([1, 3])
            with tc1:
                st.markdown("### Indicators")
                for k, v in technical.get("indicators", {}).items():
                    st.metric(k, v)
            with tc2:
                fig_price = make_price_chart(
                    prices,
                    technical.get("sma_20"),
                    technical.get("sma_50"),
                    technical.get("sma_200"),
                    ticker,
                )
                st.pyplot(fig_price, use_container_width=True)
                plt.close(fig_price)

                fig_rsi = make_rsi_chart(prices, ticker)
                st.pyplot(fig_rsi, use_container_width=True)
                plt.close(fig_rsi)
        else:
            st.warning("No price history available for technical charts.")

        st.markdown("---")
        with st.expander("📋 Detailed Technical Reasoning", expanded=(depth == "detailed")):
            for r in technical.get("reasons", []):
                st.markdown(f"- {r}")

        with st.expander("📚 Technical terms explained"):
            st.markdown("""
| Term | Explanation |
|------|-------------|
| **SMA 20 / 50 / 200** | Simple Moving Averages over 20, 50, and 200 trading days. Price above SMA = uptrend. |
| **RSI** | Relative Strength Index (0–100). >70 = overbought, <30 = oversold, 30–70 = neutral. |
| **Golden Cross** | When SMA 20 crosses above SMA 50 — bullish signal. |
| **Death Cross** | When SMA 20 crosses below SMA 50 — bearish signal. |
| **MACD** | Momentum indicator using fast & slow EMA difference. Positive = bullish momentum. |
| **Volatility** | Annualised std dev of daily returns. Higher = bigger price swings. |
""")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 3 — SENTIMENT
    # ─────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("📰 News Sentiment Agent")

        sent_color = {"Positive":"#34d399","Neutral":"#fbbf24","Negative":"#f87171"}.get(news_sen,"#94a3b8")
        sent_icon  = {"Positive":"😊","Neutral":"😐","Negative":"😟"}.get(news_sen,"")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Sentiment", f"{sent_icon} {news_sen}")
        with sc2:
            score_val = sentiment.get("sentiment_score", 0.0)
            st.metric("Polarity Score", f"{score_val:+.3f}")
        with sc3:
            st.metric("Headlines Analysed", sentiment.get("headline_count", 0))

        st.markdown(sentiment.get("summary",""))

        # Headline table
        headlines = sentiment.get("headlines", [])
        if headlines:
            st.markdown("### Headline Sentiment Breakdown")
            df_hl = {
                "Headline": [h["title"] for h in headlines],
                "Polarity": [h["polarity"] for h in headlines],
                "Label": [
                    "✅ Positive" if h["polarity"] > 0.05
                    else ("❌ Negative" if h["polarity"] < -0.05 else "➡️ Neutral")
                    for h in headlines
                ],
            }
            st.dataframe(df_hl, use_container_width=True, hide_index=True)

        if sentiment.get("data_source") == "sample":
            st.info(
                "ℹ️ **Demo Mode**: Using sample news headlines. "
                "For live analysis, configure a news API key in `.env` and set `use_api=True`."
            )
        elif sentiment.get("data_source") == "live_yfinance":
            st.success(
                "✅ **Live Mode**: Using real-time news headlines fetched directly from Yahoo Finance."
            )

        st.markdown("---")
        with st.expander("📋 Full Sentiment Reasoning", expanded=(depth == "detailed")):
            for r in sentiment.get("reasons", []):
                st.markdown(f"- {r}")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 4 — RISK
    # ─────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("⚠️ Risk Analysis Agent")

        risk_score = risk.get("risk_score", 5.0)
        risk_color = {"Low":"#34d399","Medium":"#fbbf24","High":"#f87171"}.get(risk_lv,"#94a3b8")
        risk_icon  = {"Low":"🟢","Medium":"🟡","High":"🔴"}.get(risk_lv,"⚪")

        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.markdown(
                f"<h3 style='color:{risk_color};'>{risk_icon} {risk_lv} Risk</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Score: {risk_score:.1f} / 10**")
            st.progress(risk_score / 10)
            st.markdown(risk.get("summary",""))

        with rc2:
            st.markdown("### Risk Factor Breakdown")
            rf = risk.get("risk_factors", {})
            if rf:
                for k, v in rf.items():
                    st.metric(k, v)

        st.markdown("---")
        with st.expander("📋 Detailed Risk Reasoning", expanded=(depth == "detailed")):
            for r in risk.get("reasons", []):
                st.markdown(f"- {r}")

        with st.expander("📚 Risk terms explained"):
            st.markdown("""
| Term | Explanation |
|------|-------------|
| **Volatility** | How much the stock price swings. Annualised % std dev of daily returns. |
| **Beta** | Market sensitivity. Beta 1.5 means stock moves 50% more than the index. |
| **Short Interest** | % of shares sold short. High short interest = bearish market sentiment. |
| **Max Drawdown** | Largest peak-to-trough drop in the price history. |
| **52-week Range** | Highest and lowest price over the past year — shows price band. |
""")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 5 — DECISION
    # ─────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("🏁 Decision Agent — Final Recommendation")

        dc1, dc2 = st.columns([1, 2])
        with dc1:
            rec_css_cls = {"Buy":"rec-buy","Hold":"rec-hold","Sell":"rec-sell"}.get(rec,"rec-hold")
            st.markdown(
                f"<div style='text-align:center; padding:1rem;'>"
                f"<span class='rec-badge {rec_css_cls}'>{rec_icon} {rec}</span>"
                f"<br><br><span style='color:#94a3b8; font-size:0.9rem;'>Confidence</span>"
                f"<br><span style='font-size:2rem; font-weight:700; color:#e2e8f0;'>{conf:.0f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(int(conf) / 100)

        with dc2:
            st.markdown("### Signal Breakdown")
            bd = decision.get("signal_breakdown", {})
            for agent_name, info_dict in bd.items():
                val = info_dict.get("value","")
                direction = info_dict.get("direction","Hold")
                d_icon = {"Buy":"🟢","Hold":"🟡","Sell":"🔴"}.get(direction,"⚪")
                st.markdown(f"**{agent_name}**: {val} — {d_icon} {direction}")

        st.markdown("---")
        st.markdown("### Decision Summary")
        st.success(decision.get("summary","")) if rec == "Buy" else (
            st.warning(decision.get("summary","")) if rec == "Hold" else
            st.error(decision.get("summary",""))
        )

        llm_summary = decision.get("llm_summary")
        if llm_summary:
            st.markdown("### 🤖 AI Analyst Summary")
            st.info(llm_summary)

        with st.expander("📋 Full Decision Reasoning", expanded=True):
            for r in decision.get("reasons", []):
                st.markdown(f"- {r}")

        st.markdown("---")
        st.info("⚠️ **Educational Disclaimer**: This recommendation is generated by an "
                "automated rule-based AI system for **educational purposes only**. "
                "It does **not** constitute financial advice. Always consult a qualified "
                "financial advisor before making investment decisions.")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 6 — FULL REPORT
    # ─────────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("📄 Full Research Report")
        report_text = report.get("report_text","")
        if report_text:
            st.markdown(report_text, unsafe_allow_html=False)

            st.markdown("---")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="⬇️ Download as Markdown (.md)",
                    data=report_text,
                    file_name=report.get("filename", f"{ticker}_report.md"),
                    mime="text/markdown",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    label="⬇️ Download as Text (.txt)",
                    data=report_text,
                    file_name=report.get("filename","").replace(".md",".txt") or f"{ticker}_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            if report.get("report_path"):
                st.success(f"✅ Report auto-saved to: `{report.get('report_path')}`")
        else:
            st.error("Report generation failed.")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT — trigger analysis when button clicked
# ═══════════════════════════════════════════════════════════════════════════

if run_analysis:
    if not ticker_input:
        st.error("❌ Please enter a stock ticker symbol before clicking Analyse.")
    elif not is_valid_ticker(ticker_input):
        st.error(
            f"❌ `{ticker_input}` doesn't look like a valid ticker. "
            "Use 1–10 alphanumeric characters (e.g. AAPL, BRK.B, TSLA)."
        )
    else:
        with st.spinner(f"🔄 Running 7-agent analysis for **{ticker_input}** … this may take 10–20 seconds."):
            results = run_pipeline(ticker_input)

        if "error" in results:
            st.error(results["error"])
        else:
            st.success(f"✅ Analysis complete for **{ticker_input}** — {results['data'].get('company_name','')}")
            render_dashboard(results, depth)

elif not run_analysis:
    # Show placeholder when no analysis has been run yet
    st.markdown("""
<div style="text-align:center; padding:3rem 1rem; color:#ffffff;">
    <div style="font-size:4rem;">📊</div>
    <div style="font-size:1.2rem; font-weight:600; color:#ffffff; margin-top:1rem;">
        Enter a ticker symbol above and click <strong>Analyse Stock</strong> to begin.
    </div>
    <div style="font-size:0.9rem; color:#ffffff; margin-top:0.5rem;">
        Try: AAPL · MSFT · NVDA · TSLA · GOOGL · AMZN · META
    </div>
</div>
""", unsafe_allow_html=True)
