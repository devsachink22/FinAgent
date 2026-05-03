# agents/__init__.py
# This package contains all the individual agents for the FinAgent system.

from agents.data_collection_agent import DataCollectionAgent
from agents.financial_analysis_agent import FinancialAnalysisAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.risk_analysis_agent import RiskAnalysisAgent
from agents.decision_agent import DecisionAgent
from agents.report_agent import ReportAgent

__all__ = [
    "DataCollectionAgent",
    "FinancialAnalysisAgent",
    "TechnicalAnalysisAgent",
    "NewsSentimentAgent",
    "RiskAnalysisAgent",
    "DecisionAgent",
    "ReportAgent",
]
