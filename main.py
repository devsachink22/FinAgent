"""
main.py
-------
FinAgent Command-Line Interface (CLI)
--------------------------------------
Run a full multi-agent stock analysis from the terminal.

Usage
-----
    python main.py --ticker AAPL
    python main.py --ticker MSFT --verbose
    python main.py --ticker NVDA --save-report

The final report is printed to stdout and saved to the reports/ folder.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────
if hasattr(sys.stdout, "buffer") and (not sys.stdout.encoding or sys.stdout.encoding.lower() != "utf-8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and (not sys.stderr.encoding or sys.stderr.encoding.lower() != "utf-8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Ensure project root is on sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Load environment variables (optional API keys) ────────────────────────
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv is optional

from agents.data_collection_agent import DataCollectionAgent
from agents.financial_analysis_agent import FinancialAnalysisAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.risk_analysis_agent import RiskAnalysisAgent
from agents.decision_agent import DecisionAgent
from agents.report_agent import ReportAgent
from utils.helpers import is_valid_ticker


# ── Logging setup ─────────────────────────────────────────────────────────
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Pretty console helpers ────────────────────────────────────────────────
def print_header(text: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_section(title: str, body: str = "") -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    if body:
        print(body)


def print_bullet(items: list, indent: int = 2) -> None:
    for item in items:
        print(" " * indent + str(item))


# ── Main pipeline ─────────────────────────────────────────────────────────
def run_pipeline(ticker: str, verbose: bool = False) -> dict:
    """
    Execute the full multi-agent FinAgent pipeline.

    Returns a dict containing all agent results.
    """
    ticker = ticker.strip().upper()

    print_header(f"FinAgent — Analysing: {ticker}")
    print("⚠️  Educational purposes only. Not financial advice.\n")

    results: dict = {}

    # ── Agent 1: Data Collection ──────────────────────────────────────────
    print("📡  [1/7] Data Collection Agent running ...", end=" ", flush=True)
    t0 = time.perf_counter()
    data_agent = DataCollectionAgent()
    data = data_agent.run(ticker)
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    if not data.get("success"):
        print(f"\n{data.get('error', 'Unknown error in data collection.')}")
        sys.exit(1)

    results["data"] = data
    print(f"     Company : {data.get('company_name', ticker)}")
    print(f"     Price   : ${data.get('current_price', 'N/A')}")

    # ── Agent 2: Financial Analysis ───────────────────────────────────────
    print("💰  [2/7] Financial Analysis Agent running ...", end=" ", flush=True)
    t0 = time.perf_counter()
    fin_agent = FinancialAnalysisAgent()
    financial = fin_agent.run(data)
    print(f"done ({time.perf_counter()-t0:.1f}s)")
    results["financial"] = financial
    print(f"     Score   : {financial.get('financial_score', 'N/A')}/10 "
          f"({financial.get('score_label', '')})")

    # ── Agent 3: Technical Analysis ───────────────────────────────────────
    print("📈  [3/7] Technical Analysis Agent running ...", end=" ", flush=True)
    t0 = time.perf_counter()
    tech_agent = TechnicalAnalysisAgent()
    technical = tech_agent.run(data)
    print(f"done ({time.perf_counter()-t0:.1f}s)")
    results["technical"] = technical
    print(f"     Signal  : {technical.get('signal', 'N/A')}")

    # ── Agent 4: News Sentiment ───────────────────────────────────────────
    print("📰  [4/7] News Sentiment Agent running ...", end=" ", flush=True)
    t0 = time.perf_counter()
    news_agent = NewsSentimentAgent()
    sentiment = news_agent.run(data)
    print(f"done ({time.perf_counter()-t0:.1f}s)")
    results["sentiment"] = sentiment
    print(f"     Sent.   : {sentiment.get('sentiment', 'N/A')} "
          f"({sentiment.get('sentiment_score', 0.0):+.3f})")

    # ── Agent 5: Risk Analysis ────────────────────────────────────────────
    print("⚠️   [5/7] Risk Analysis Agent running ...", end=" ", flush=True)
    t0 = time.perf_counter()
    risk_agent = RiskAnalysisAgent()
    risk = risk_agent.run(data, technical, sentiment, financial)
    print(f"done ({time.perf_counter()-t0:.1f}s)")
    results["risk"] = risk
    print(f"     Risk    : {risk.get('risk_level', 'N/A')} "
          f"({risk.get('risk_score', 'N/A')}/10)")

    # ── Agent 6: Decision ─────────────────────────────────────────────────
    print("🏁  [6/7] Decision Agent running ...", end=" ", flush=True)
    t0 = time.perf_counter()
    dec_agent = DecisionAgent()
    decision = dec_agent.run(financial, technical, sentiment, risk, data)
    print(f"done ({time.perf_counter()-t0:.1f}s)")
    results["decision"] = decision
    rec = decision.get("recommendation", "N/A")
    conf = decision.get("confidence", 0.0)
    print(f"     Rec.    : {rec} ({conf:.0f}% confidence)")

    # ── Agent 7: Report ───────────────────────────────────────────────────
    print("📄  [7/7] Report Agent generating report ...", end=" ", flush=True)
    t0 = time.perf_counter()
    rep_agent = ReportAgent()
    report = rep_agent.run(data, financial, technical, sentiment, risk, decision)
    print(f"done ({time.perf_counter()-t0:.1f}s)")
    results["report"] = report

    return results


def print_results(results: dict, verbose: bool = False) -> None:
    """Pretty-print the pipeline results to the terminal."""
    data = results.get("data", {})
    financial = results.get("financial", {})
    technical = results.get("technical", {})
    sentiment = results.get("sentiment", {})
    risk = results.get("risk", {})
    decision = results.get("decision", {})
    report = results.get("report", {})

    print_section(f"📊 FINAL REPORT — {data.get('company_name', '')} ({data.get('ticker', '')})")

    rec = decision.get("recommendation", "N/A")
    conf = decision.get("confidence", 0.0)
    emoji = {"Buy": "🟢", "Hold": "🟡", "Sell": "🔴"}.get(rec, "⚪")

    print(f"\n  {emoji} Recommendation : {rec}")
    print(f"  📊 Confidence    : {conf:.0f}%")
    print(f"  💰 Financial     : {financial.get('financial_score', 'N/A')}/10 ({financial.get('score_label', '')})")
    print(f"  📈 Technical     : {technical.get('signal', 'N/A')}")
    print(f"  📰 Sentiment     : {sentiment.get('sentiment', 'N/A')}")
    print(f"  ⚠️  Risk Level    : {risk.get('risk_level', 'N/A')}")

    if verbose:
        print_section("Decision Reasoning")
        print_bullet(decision.get("reasons", []))

        print_section("Financial Reasoning")
        print_bullet(financial.get("reasons", []))

        print_section("Technical Reasoning")
        print_bullet(technical.get("reasons", []))

        print_section("Sentiment Reasoning")
        print_bullet(sentiment.get("reasons", []))

        print_section("Risk Reasoning")
        print_bullet(risk.get("reasons", []))

    print_section("Report")
    if report.get("report_path"):
        print(f"  ✅ Report saved to: {report.get('report_path')}")
    else:
        print("  ⚠️  Report could not be saved to disk.")

    print("\n" + "=" * 60)
    print("  ⚠️  DISCLAIMER: Educational purposes only.")
    print("      This is NOT financial advice.")
    print("=" * 60 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="finagent",
        description=(
            "FinAgent: Multi-Agent Stock Research System\n"
            "Educational purposes only — NOT financial advice."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT, TSLA, NVDA)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed reasoning from each agent",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Save the report to the reports/ folder (default: True)",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    # Validate ticker
    if not is_valid_ticker(args.ticker):
        print(
            f"❌ '{args.ticker}' does not look like a valid ticker symbol.\n"
            "   Tickers are 1–10 alphanumeric characters (e.g. AAPL, BRK.B)."
        )
        sys.exit(1)

    # Run pipeline
    try:
        results = run_pipeline(args.ticker, verbose=args.verbose)
        print_results(results, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n❌ Unexpected error: {exc}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
