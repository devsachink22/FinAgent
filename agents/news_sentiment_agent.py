"""
agents/news_sentiment_agent.py
--------------------------------
News Sentiment Agent — analyses headlines to gauge market sentiment.

Two operating modes
-------------------
1. Basic (default): loads sample headlines from data/sample_news.json.
   Works entirely offline and requires no paid API.
2. Optional API mode: stub is included for easy future extension with a
   real news API (e.g., NewsAPI, Alpha Vantage News).

Sentiment engine
----------------
Uses TextBlob for lightweight polarity scoring.
VaderSentiment is used as an alternative if TextBlob is unavailable.
Falls back to a simple keyword rule system if neither library is present.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Path to the bundled sample news data (relative to project root)
_SAMPLE_NEWS_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_news.json"


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

def _textblob_sentiment(text: str) -> float:
    """Return polarity in [-1, +1] using TextBlob."""
    try:
        from textblob import TextBlob  # type: ignore
        return TextBlob(text).sentiment.polarity
    except ImportError:
        raise


def _vader_sentiment(text: str) -> float:
    """Return compound score in [-1, +1] using VaderSentiment."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)["compound"]
    except ImportError:
        raise


def _keyword_sentiment(text: str) -> float:
    """
    Rule-based fallback: count positive/negative finance keywords.
    Returns a value in [-1, +1].
    """
    positive_words = {
        "surge", "rally", "beat", "growth", "profit", "gain", "record",
        "strong", "bullish", "outperform", "upgrade", "positive", "rise",
        "soar", "jump", "breakthrough", "expansion", "revenue growth",
        "dividend increase", "buy", "accumulate", "opportunity",
    }
    negative_words = {
        "fall", "drop", "miss", "loss", "decline", "weak", "bearish",
        "downgrade", "negative", "slump", "crash", "concern", "risk",
        "layoff", "lawsuit", "investigation", "debt", "shortfall",
        "recession", "warning", "sell", "reduce", "avoid", "underperform",
    }
    lower = text.lower()
    pos = sum(1 for w in positive_words if w in lower)
    neg = sum(1 for w in negative_words if w in lower)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def score_headline(text: str) -> float:
    """
    Score a single headline, trying TextBlob → VADER → keyword fallback.
    Returns polarity in [-1, +1].
    """
    for fn in (_textblob_sentiment, _vader_sentiment, _keyword_sentiment):
        try:
            return fn(text)
        except (ImportError, Exception):
            continue
    return 0.0


def polarity_to_label(polarity: float) -> str:
    if polarity > 0.05:
        return "Positive"
    if polarity < -0.05:
        return "Negative"
    return "Neutral"


# ---------------------------------------------------------------------------
# News Sentiment Agent
# ---------------------------------------------------------------------------

class NewsSentimentAgent:
    """
    Analyses news headlines for a stock ticker.

    Usage
    -----
    agent  = NewsSentimentAgent()
    result = agent.run(data_collection_result)
    """

    def __init__(self, use_api: bool = False, api_key: Optional[str] = None) -> None:
        self.name = "News Sentiment Agent"
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        self.use_api = use_api or bool(self.api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform news sentiment analysis.

        Parameters
        ----------
        data : dict returned by DataCollectionAgent.run()

        Returns
        -------
        dict with keys:
            success         : bool
            error           : str | None
            sentiment       : str   ('Positive' | 'Neutral' | 'Negative')
            sentiment_score : float (average polarity, –1 to +1)
            headline_count  : int
            headlines       : list[dict]  each with 'title' and 'polarity'
            summary         : str
            reasons         : list[str]
            data_source     : str  ('sample' | 'api')
        """
        result: Dict[str, Any] = {
            "success": False,
            "error": None,
            "sentiment": "Neutral",
            "sentiment_score": 0.0,
            "headline_count": 0,
            "headlines": [],
            "summary": "",
            "reasons": [],
            "data_source": "sample",
        }

        if not data.get("success"):
            result["error"] = "Data collection failed — cannot run sentiment analysis."
            return result

        ticker: str = data.get("ticker", "N/A")
        company: str = data.get("company_name", ticker)
        logger.info("[%s] Running sentiment analysis for %s", self.name, ticker)

        # ── Load headlines ─────────────────────────────────────────────────
        if self.use_api and self.api_key:
            headlines = self._fetch_from_api(ticker)
            result["data_source"] = "api"
        else:
            live_news = data.get("news")
            if live_news and isinstance(live_news, list) and len(live_news) > 0:
                headlines = live_news
                result["data_source"] = "live_yfinance"
            else:
                headlines = self._load_sample_news(ticker)
                result["data_source"] = "sample"

        if not headlines:
            result["error"] = (
                f"No news headlines available for {ticker}. "
                "Using neutral sentiment as default."
            )
            result["success"] = True
            result["summary"] = f"No news data available for {company}. Sentiment set to Neutral."
            return result

        # ── Score each headline ────────────────────────────────────────────
        scored: List[Dict[str, Any]] = []
        for h in headlines:
            title = h.get("title", "") if isinstance(h, dict) else str(h)
            polarity = score_headline(title)
            scored.append({"title": title, "polarity": round(polarity, 3)})

        avg_polarity = sum(s["polarity"] for s in scored) / len(scored)
        sentiment = polarity_to_label(avg_polarity)

        # ── Build reasons ──────────────────────────────────────────────────
        reasons: List[str] = []
        pos = sum(1 for s in scored if s["polarity"] > 0.05)
        neg = sum(1 for s in scored if s["polarity"] < -0.05)
        neu = len(scored) - pos - neg

        reasons.append(
            f"📰 Analysed {len(scored)} headline(s): "
            f"{pos} positive, {neu} neutral, {neg} negative."
        )

        if sentiment == "Positive":
            reasons.append(
                f"✅ Overall sentiment is Positive (avg polarity: {avg_polarity:+.3f}). "
                "News flow appears favourable for the stock."
            )
        elif sentiment == "Negative":
            reasons.append(
                f"❌ Overall sentiment is Negative (avg polarity: {avg_polarity:+.3f}). "
                "Negative news may create selling pressure."
            )
        else:
            reasons.append(
                f"ℹ️ Overall sentiment is Neutral (avg polarity: {avg_polarity:+.3f}). "
                "No strong directional bias from news."
            )

        if result["data_source"] == "sample":
            reasons.append(
                "⚠️ Using sample/demo news headlines — not real-time market news. "
                "For live analysis, integrate a news API."
            )
        elif result["data_source"] == "live_yfinance":
            reasons.append(
                "✅ Using live recent news headlines fetched from Yahoo Finance."
            )

        # Top 3 positive / negative examples
        top_pos = sorted(scored, key=lambda x: x["polarity"], reverse=True)[:3]
        top_neg = sorted(scored, key=lambda x: x["polarity"])[:3]
        for h in top_pos:
            if h["polarity"] > 0.05:
                reasons.append(f"  📈 \"{h['title']}\" (score: {h['polarity']:+.3f})")
        for h in top_neg:
            if h["polarity"] < -0.05:
                reasons.append(f"  📉 \"{h['title']}\" (score: {h['polarity']:+.3f})")

        # ── Summary ────────────────────────────────────────────────────────
        summary = (
            f"News sentiment for {company} is **{sentiment}** "
            f"based on {len(scored)} headline(s). "
            f"Average polarity score: {avg_polarity:+.3f}."
        )

        result.update(
            {
                "success": True,
                "sentiment": sentiment,
                "sentiment_score": round(avg_polarity, 4),
                "headline_count": len(scored),
                "headlines": scored,
                "summary": summary,
                "reasons": reasons,
            }
        )
        logger.info("[%s] %s — sentiment: %s", self.name, ticker, sentiment)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_sample_news(self, ticker: str) -> List[dict]:
        """
        Load headlines from data/sample_news.json.

        The file may contain ticker-specific headlines under the ticker key,
        or a generic 'DEFAULT' list used for any unknown ticker.
        """
        try:
            with open(_SAMPLE_NEWS_PATH, "r", encoding="utf-8") as f:
                news_data: dict = json.load(f)

            # Try ticker-specific, then DEFAULT
            headlines = (
                news_data.get(ticker.upper())
                or news_data.get("DEFAULT")
                or []
            )
            return headlines  # type: ignore[return-value]

        except FileNotFoundError:
            logger.warning("[%s] sample_news.json not found at %s", self.name, _SAMPLE_NEWS_PATH)
            return self._builtin_fallback_headlines()
        except json.JSONDecodeError as exc:
            logger.error("[%s] JSON parse error in sample_news.json: %s", self.name, exc)
            return self._builtin_fallback_headlines()

    @staticmethod
    def _builtin_fallback_headlines() -> List[dict]:
        """Hardcoded minimal fallback in case the JSON file is missing."""
        return [
            {"title": "Markets show mixed signals amid economic uncertainty"},
            {"title": "Analysts divided on near-term stock outlook"},
            {"title": "Investors watch for quarterly earnings results"},
        ]

    def _fetch_from_api(self, ticker: str) -> List[dict]:
        """
        Fetch live news headlines from NewsAPI.
        """
        import requests
        if not self.api_key:
            return self._load_sample_news(ticker)

        logger.info("[%s] Fetching live news for %s via NewsAPI...", self.name, ticker)
        try:
            url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&apiKey={self.api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                if articles:
                    return [{"title": a.get("title", "")} for a in articles[:10]]
            else:
                logger.warning("[%s] NewsAPI returned status %s: %s", self.name, response.status_code, response.text)
        except Exception as exc:
            logger.error("[%s] Error fetching from NewsAPI: %s", self.name, exc)

        logger.info("[%s] Falling back to sample news.", self.name)
        return self._load_sample_news(ticker)
