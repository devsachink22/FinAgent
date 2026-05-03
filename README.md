# 📊 FinAgent — Multi-Agent Stock Research and Recommendation System

> **⚠️ Educational Disclaimer**: This project is built for **educational and research purposes only**. It does **not** constitute financial advice, investment recommendations, or any form of professional guidance. Always consult a qualified financial advisor before making investment decisions.

---

## 📌 Project Overview

**FinAgent** is a Python-based multi-agent AI system paired with a modern Streamlit finance dashboard. It analyses a stock ticker using financial data, technical indicators, news sentiment, and risk assessment — then generates an explainable **Buy / Hold / Sell** recommendation.

This project demonstrates:
- **Multi-agent architecture** with clean separation of concerns
- **Rule-based AI reasoning** without any paid API requirement
- **End-to-end pipeline** from data collection to report generation
- **Professional Streamlit dashboard** suitable for a graduate-level AI/Finance portfolio

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **7 Specialised Agents** | Each agent handles one domain independently |
| 📊 **Financial Analysis** | P/E, margins, ROE, debt, growth scoring |
| 📈 **Technical Analysis** | SMA, RSI, MACD, volatility, momentum |
| 📰 **Sentiment Analysis** | NLP on news headlines (TextBlob) |
| ⚠️ **Risk Assessment** | Multi-factor risk scoring |
| 🏁 **Decision Engine** | Rule-based Buy/Hold/Sell with confidence |
| 📄 **Report Generator** | Full Markdown research report |
| 🖥️ **Streamlit Dashboard** | Dark-mode finance dashboard with charts |
| 💻 **CLI Interface** | Terminal analysis with `--ticker` flag |

---

## 🏗️ Multi-Agent Architecture

```
User Input (Ticker)
       │
       ▼
┌─────────────────────┐
│  Data Collection    │  ← yfinance (prices, fundamentals)
│       Agent         │
└────────┬────────────┘
         │ stock_data
    ┌────┴─────────────────────────────────┐
    │            │            │            │
    ▼            ▼            ▼            ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│Fin.    │  │Tech.   │  │News    │  │(passed │
│Analysis│  │Analysis│  │Sentim. │  │through)│
│Agent   │  │Agent   │  │Agent   │  │        │
└───┬────┘  └───┬────┘  └───┬────┘  └────────┘
    │            │            │
    └─────┬──────┘────────────┘
          │ (all results)
          ▼
    ┌───────────┐
    │   Risk    │
    │ Analysis  │
    │   Agent   │
    └─────┬─────┘
          │
          ▼
    ┌───────────┐
    │ Decision  │
    │   Agent   │  → Buy / Hold / Sell + Confidence
    └─────┬─────┘
          │
          ▼
    ┌───────────┐
    │  Report   │
    │   Agent   │  → Markdown report saved to reports/
    └───────────┘
```

---

## 📁 Folder Structure

```
finagent/
│
├── app.py                      # Streamlit dashboard
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env.example                # Environment variable template
│
├── agents/
│   ├── __init__.py
│   ├── data_collection_agent.py     # yfinance data fetching
│   ├── financial_analysis_agent.py  # Fundamental analysis + scoring
│   ├── technical_analysis_agent.py  # SMA, RSI, MACD, volatility
│   ├── news_sentiment_agent.py      # Headline NLP (TextBlob)
│   ├── risk_analysis_agent.py       # Multi-factor risk scoring
│   ├── decision_agent.py            # Final Buy/Hold/Sell logic
│   └── report_agent.py             # Markdown report generator
│
├── utils/
│   ├── __init__.py
│   ├── indicators.py           # RSI, SMA, volatility, MACD, drawdown
│   ├── scoring.py              # Financial health & risk scoring rules
│   └── helpers.py              # Formatting, validation utilities
│
├── data/
│   └── sample_news.json        # Demo headlines for AAPL, MSFT, TSLA, etc.
│
└── reports/
    └── sample_report.md        # Example output report
```

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
cd finagent
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download TextBlob corpora (one-time)

```bash
python -m textblob.download_corpora
```

### 5. (Optional) Configure API keys

```bash
copy .env.example .env   # Windows
cp .env.example .env      # macOS/Linux
# Edit .env and add optional keys
```

---

## 🚀 How to Run

### A. Streamlit Web Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### B. Command-Line Interface

```bash
# Basic analysis
python main.py --ticker AAPL

# Verbose mode (shows all agent reasoning)
python main.py --ticker MSFT --verbose

# Other examples
python main.py --ticker NVDA
python main.py --ticker TSLA --verbose
```

---

## 📊 Example Output

```
============================================================
  FinAgent — Analysing: AAPL
============================================================
⚠️  Educational purposes only. Not financial advice.

📡  [1/7] Data Collection Agent running ...    done (2.3s)
     Company : Apple Inc.
     Price   : $195.12
💰  [2/7] Financial Analysis Agent running ... done (0.0s)
     Score   : 7.2/10 (Strong)
📈  [3/7] Technical Analysis Agent running ... done (0.0s)
     Signal  : Bullish
📰  [4/7] News Sentiment Agent running ...     done (0.0s)
     Sent.   : Positive (+0.214)
⚠️   [5/7] Risk Analysis Agent running ...     done (0.0s)
     Risk    : Medium (5.2/10)
🏁  [6/7] Decision Agent running ...           done (0.0s)
     Rec.    : Buy (74% confidence)
📄  [7/7] Report Agent generating report ...   done (0.0s)

──────────────────────────────────────────────────────────
  📊 FINAL REPORT — Apple Inc. (AAPL)
──────────────────────────────────────────────────────────

  🟢 Recommendation : Buy
  📊 Confidence    : 74%
  💰 Financial     : 7.2/10 (Strong)
  📈 Technical     : Bullish
  📰 Sentiment     : Positive
  ⚠️  Risk Level    : Medium
```

---

## 🛠 Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| `yfinance` | ≥0.2.38 | Stock data & company fundamentals |
| `pandas` | ≥2.0 | Data manipulation |
| `numpy` | ≥1.26 | Numerical calculations |
| `matplotlib` | ≥3.8 | Price & RSI charts |
| `textblob` | ≥0.18 | Sentiment NLP |
| `streamlit` | ≥1.35 | Web dashboard |
| `python-dotenv` | ≥1.0 | Environment variable loading |

---

## ⚠️ Limitations

- **Not real financial advice** — rule-based scoring, not professional analysis.
- **News is sample data** — live headlines require a News API key.
- **yfinance data delays** — prices may be delayed 15–20 minutes.
- **No backtesting** — recommendations are not validated against historical performance.
- **No portfolio context** — analysis is single-stock only.
- **LLM reasoning not included** — purely deterministic rule-based logic.

---

## 🔮 Future Improvements

1. **Real-time News API** — Integrate NewsAPI or Bloomberg for live headlines.
2. **LLM-based Reasoning** — Use GPT-4 or Claude for natural-language explanations.
3. **RAG over Earnings Reports** — Retrieve and analyse 10-Q / 10-K filings.
4. **Portfolio-Level Analysis** — Analyse a basket of stocks together.
5. **Backtesting Engine** — Validate recommendations against historical data.
6. **Advanced Risk Metrics** — Sharpe ratio, VaR, Sortino ratio.
7. **Sector Comparison** — Compare a stock against its sector peers.
8. **Alerting System** — Email or Slack alerts when signals change.
9. **Multi-language Support** — Sentiment analysis in non-English headlines.
10. **Database Storage** — Persist analysis history for trend tracking.

---

## 📄 License

This project is for **educational use only**. No commercial or investment use is permitted.

---

*Built with ❤️ for graduate-level AI/Finance coursework.*
