"""Portfolio-grade equity research toolkit designed for Google Colab."""

from .config import ProjectConfig
from .data import YahooFinanceClient, SECClient, FredClient, MarketDataBundle
from .features import add_market_features
from .fundamentals import FundamentalAnalyzer, FundamentalSnapshot
from .valuation import DCFModel, DCFInputs, DCFResult, ComparableValuation
from .risk import RiskAnalyzer, PortfolioOptimizer
from .pipeline import EquityResearchPipeline, ResearchResult
from .reporting import ReportBuilder

__all__ = [
    "ProjectConfig",
    "YahooFinanceClient",
    "SECClient",
    "FredClient",
    "MarketDataBundle",
    "add_market_features",
    "FundamentalAnalyzer",
    "FundamentalSnapshot",
    "DCFModel",
    "DCFInputs",
    "DCFResult",
    "ComparableValuation",
    "RiskAnalyzer",
    "PortfolioOptimizer",
    "EquityResearchPipeline",
    "ResearchResult",
    "ReportBuilder",
]
