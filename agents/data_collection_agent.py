"""
agents/data_collection_agent.py
--------------------------------
Data Collection Agent — the first agent in the FinAgent pipeline.

Responsibilities
----------------
• Accept a stock ticker symbol from the user.
• Download company information and historical price data via yfinance.
• Return a clean, structured result dictionary consumed by downstream agents.
• Handle invalid tickers and missing data gracefully.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import os
import pandas as pd
import requests
import yfinance as yf

from utils.helpers import safe_get

logger = logging.getLogger(__name__)


class DataCollectionAgent:
    """
    Collects fundamental company data and historical OHLCV prices.

    Usage
    -----
    agent = DataCollectionAgent()
    result = agent.run("AAPL")
    """

    # Maximum history requested (5 years gives enough data for 200-day MA)
    _HISTORY_PERIOD = "5y"

    def __init__(self) -> None:
        self.name = "Data Collection Agent"
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, ticker: str) -> Dict[str, Any]:
        """
        Main entry point.

        Parameters
        ----------
        ticker : str   e.g. "AAPL", "MSFT", "TSLA"

        Returns
        -------
        dict with keys:
            success       : bool
            error         : str | None
            ticker        : str (upper-cased)
            info          : dict  (yfinance Ticker.info)
            history       : pd.DataFrame | None
            current_price : float | None
            company_name  : str
            sector        : str
            industry      : str
            market_cap    : float | None
            pe_ratio      : float | None
            revenue_growth: float | None
            profit_margins: float | None
            debt_to_equity: float | None
        """
        ticker = ticker.strip().upper()
        logger.info("[%s] Starting data collection for %s", self.name, ticker)

        result: Dict[str, Any] = {
            "success": False,
            "error": None,
            "ticker": ticker,
            "info": {},
            "history": None,
            "current_price": None,
            "company_name": ticker,
            "sector": "Unknown",
            "industry": "Unknown",
            "market_cap": None,
            "pe_ratio": None,
            "revenue_growth": None,
            "profit_margins": None,
            "debt_to_equity": None,
        }

        # ── Step 1: Fetch company info ─────────────────────────────────────
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # yfinance returns a minimal dict (e.g. {"symbol": ticker})
            # for invalid tickers instead of raising an exception.
            if not self._is_valid_info(info, ticker):
                result["error"] = (
                    f"❌ Could not find data for ticker '{ticker}'. "
                    "Please check the symbol and try again. "
                    "Examples of valid tickers: AAPL, MSFT, TSLA, NVDA, GOOGL."
                )
                return result

            result["info"] = info

        except Exception as exc:
            logger.error("[%s] Failed to fetch info for %s: %s", self.name, ticker, exc)
            result["error"] = (
                f"❌ An error occurred while fetching data for '{ticker}': {exc}. "
                "Please check your internet connection and try again."
            )
            return result

        # ── Step 2: Extract key fundamental fields ─────────────────────────
        result["company_name"] = (
            safe_get(info, "longName")
            or safe_get(info, "shortName")
            or ticker
        )
        result["sector"] = safe_get(info, "sector") or "Unknown"
        result["industry"] = safe_get(info, "industry") or "Unknown"
        result["market_cap"] = safe_get(info, "marketCap")
        result["pe_ratio"] = safe_get(info, "trailingPE") or safe_get(info, "forwardPE")
        result["revenue_growth"] = safe_get(info, "revenueGrowth")
        result["profit_margins"] = safe_get(info, "profitMargins")
        result["debt_to_equity"] = safe_get(info, "debtToEquity")

        # Current price: prefer regularMarketPrice, fall back to currentPrice
        current_price = (
            safe_get(info, "regularMarketPrice")
            or safe_get(info, "currentPrice")
            or safe_get(info, "previousClose")
        )
        result["current_price"] = float(current_price) if current_price else None

        # ── Step 3: Fetch historical OHLCV data ────────────────────────────
        try:
            history = stock.history(period=self._HISTORY_PERIOD, auto_adjust=True)

            if history is None or history.empty:
                # Fall back to 1-year data
                history = stock.history(period="1y", auto_adjust=True)

            if history is not None and not history.empty:
                result["history"] = history
                # If current_price is still missing, take last close
                if result["current_price"] is None:
                    result["current_price"] = float(history["Close"].iloc[-1])
            else:
                logger.warning("[%s] No historical data for %s", self.name, ticker)

        except Exception as exc:
            logger.warning(
                "[%s] Could not download history for %s: %s", self.name, ticker, exc
            )
            # Non-fatal: downstream agents will handle None history

        # ── Step 4: Fetch recent news ──────────────────────────────────────
        try:
            news_items = stock.news
            if news_items:
                extracted_news = []
                for item in news_items:
                    if "title" in item:
                        extracted_news.append({"title": item["title"]})
                    elif "content" in item and "title" in item["content"]:
                        extracted_news.append({"title": item["content"]["title"]})
                result["news"] = extracted_news
        except Exception as exc:
            logger.warning(
                "[%s] Could not fetch news for %s: %s", self.name, ticker, exc
            )

        result["success"] = True

        # ── Step 5: Supplement with Alpha Vantage ──────────────────────────
        if self.alpha_vantage_key:
            logger.info("[%s] Fetching Alpha Vantage data for %s...", self.name, ticker)
            try:
                # 1. Fetch Global Quote
                quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={self.alpha_vantage_key}"
                q_resp = requests.get(quote_url, timeout=10)
                if q_resp.status_code == 200:
                    q_data = q_resp.json()
                    global_quote = q_data.get("Global Quote", {})
                    if global_quote and "05. price" in global_quote:
                        result["current_price"] = float(global_quote["05. price"])
                        result["alpha_vantage_used"] = True

                # 2. Fetch Overview
                overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.alpha_vantage_key}"
                o_resp = requests.get(overview_url, timeout=10)
                if o_resp.status_code == 200:
                    o_data = o_resp.json()
                    if o_data and "Symbol" in o_data:
                        if o_data.get("PERatio") and o_data["PERatio"] != "None":
                            result["pe_ratio"] = float(o_data["PERatio"])
                        if o_data.get("ProfitMargin") and o_data["ProfitMargin"] != "None":
                            result["profit_margins"] = float(o_data["ProfitMargin"])
                        if o_data.get("MarketCapitalization") and o_data["MarketCapitalization"] != "None":
                            result["market_cap"] = float(o_data["MarketCapitalization"])
                        if o_data.get("QuarterlyRevenueGrowthYOY") and o_data["QuarterlyRevenueGrowthYOY"] != "None":
                            result["revenue_growth"] = float(o_data["QuarterlyRevenueGrowthYOY"])
                        
                        result["alpha_vantage_overview_used"] = True
            except Exception as exc:
                logger.warning("[%s] Error fetching Alpha Vantage data: %s", self.name, exc)

        logger.info(
            "[%s] Collected data for %s (%s). Price: %s",
            self.name,
            ticker,
            result["company_name"],
            result["current_price"],
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_info(info: dict, ticker: str) -> bool:
        """
        Return True if the info dict looks like a real stock, not an error stub.
        yfinance returns {'symbol': ticker, 'quoteType': 'NONE'} for bad tickers.
        """
        if not info:
            return False

        # A valid response has at least one of these fields
        valid_fields = [
            "longName", "shortName", "currentPrice",
            "regularMarketPrice", "previousClose", "marketCap",
        ]
        has_data = any(info.get(f) is not None for f in valid_fields)

        # Reject stub responses
        quote_type = info.get("quoteType", "")
        if quote_type in ("NONE", ""):
            # Could still be valid (some ETFs have empty quoteType)
            return has_data

        return True
