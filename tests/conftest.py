"""
Pytest Configuration and Central Fixtures

This file provides reusable test fixtures for the AI Trading Demo test suite.
Fixtures are used to reduce code duplication and provide consistent test data
across all test modules following the DRY principle.

Author: AI Trading Demo Team
Version: 2.0 (AI-Powered Architecture)
"""

import pytest
import pandas as pd
import datetime
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

# Import our modules for testing
from core.config import AITradingConfig, get_config, reset_config
from core.data_manager import DataManager
from core.strategy import TradingStrategy
from core.news_fetcher import NewsFetcher, NewsArticle
from core.ai_analyzer import AIAnalyzer, AIAnalysisResult


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """A reusable fixture providing OHLCV stock data for testing.

    This fixture loads the sample CSV file and returns a pandas DataFrame
    with proper date formatting for testing various components.

    Returns:
        pd.DataFrame: Sample OHLCV data with Date, Open, High, Low, Close, Volume columns.
    """
    # Load the sample CSV file
    csv_path = Path(__file__).parent / "fixtures" / "sample_ohlcv.csv"
    df = pd.read_csv(csv_path)

    # Convert Date column to proper date format
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    return df


@pytest.fixture
def sample_minimal_data() -> pd.DataFrame:
    """A minimal dataset for testing edge cases.

    Returns:
        pd.DataFrame: Minimal OHLCV data with just 5 rows.
    """
    data = {
        "Date": [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
            datetime.date(2023, 1, 4),
            datetime.date(2023, 1, 5),
        ],
        "Open": [100.0, 102.0, 104.0, 106.0, 108.0],
        "High": [105.0, 106.0, 108.0, 110.0, 112.0],
        "Low": [99.0, 101.0, 103.0, 105.0, 107.0],
        "Close": [102.0, 104.0, 106.0, 108.0, 110.0],
        "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_with_indicators() -> pd.DataFrame:
    """Sample data with pre-calculated SMA indicators.

    This fixture provides data with SMA20 and SMA50 already calculated,
    useful for testing signal generation without indicator calculation.

    Returns:
        pd.DataFrame: OHLCV data with SMA20 and SMA50 columns.
    """
    # Create base data for testing crossover scenarios
    dates = pd.date_range(start="2023-01-01", periods=60, freq="D")

    data = {
        "Date": [date.date() for date in dates],
        "Open": [100.0 + i for i in range(60)],
        "High": [105.0 + i for i in range(60)],
        "Low": [95.0 + i for i in range(60)],
        "Close": [100.0 + i for i in range(60)],
        "Volume": [1000000 + i * 10000 for i in range(60)],
    }
    df = pd.DataFrame(data)

    # Note: In v2 AI-powered version, indicators are calculated by the strategy internally

    return df


@pytest.fixture
def crossover_scenario_data() -> pd.DataFrame:
    """Fixture providing data specifically designed to test crossover scenarios.

    This fixture creates data where SMA20 crosses both above and below SMA50
    at specific points to test signal generation logic.

    Returns:
        pd.DataFrame: Data designed for testing buy and sell signals.
    """
    # Create scenario where SMA20 starts below SMA50, crosses above, then crosses below
    data = {
        "Date": [
            datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(55)
        ],
        "Open": [150.0] * 55,
        "High": [155.0] * 55,
        "Low": [145.0] * 55,
        "Close": [],
        "Volume": [1000000] * 55,
        "SMA20": [],
        "SMA50": [],
    }

    # Design specific close prices and SMA values to create crossovers
    for i in range(55):
        if i < 25:
            # SMA20 below SMA50 (bearish)
            close = 150.0 - (25 - i) * 0.5  # Gradually increasing
            sma20 = close - 2.0
            sma50 = close + 1.0
        elif i == 25:
            # Crossover point - SMA20 crosses above SMA50 (BUY SIGNAL)
            close = 150.0
            sma20 = 150.5  # Above SMA50
            sma50 = 150.0  # Below SMA20
        elif i < 40:
            # SMA20 above SMA50 (bullish)
            close = 150.0 + (i - 25) * 0.3
            sma20 = close + 1.0
            sma50 = close - 1.0
        elif i == 40:
            # Crossover point - SMA20 crosses below SMA50 (SELL SIGNAL)
            close = 155.0
            sma20 = 154.5  # Below SMA50
            sma50 = 155.0  # Above SMA20
        else:
            # SMA20 below SMA50 again (bearish)
            close = 155.0 - (i - 40) * 0.2
            sma20 = close - 1.0
            sma50 = close + 0.5

        data["Close"].append(close)
        data["SMA20"].append(sma20)
        data["SMA50"].append(sma50)

    return pd.DataFrame(data)


@pytest.fixture
def mock_yfinance_success():
    """Mock yfinance to return successful data.

    This fixture mocks the yfinance.Ticker.history method to return
    predefined successful data, avoiding real API calls during testing.
    """
    mock_data = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 104.0],
            "High": [105.0, 106.0, 108.0],
            "Low": [99.0, 101.0, 103.0],
            "Close": [102.0, 104.0, 106.0],
            "Volume": [1000000, 1100000, 1200000],
        }
    )
    mock_data.index = pd.date_range("2023-01-01", periods=3, freq="D")
    mock_data.index.name = "Date"

    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_instance
        yield mock_ticker


@pytest.fixture
def mock_yfinance_empty():
    """Mock yfinance to return empty data.

    This fixture simulates the case where yfinance returns no data,
    useful for testing error handling.
    """
    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker.return_value = mock_instance
        yield mock_ticker


@pytest.fixture
def mock_yfinance_error():
    """Mock yfinance to raise an exception.

    This fixture simulates network or API errors from yfinance,
    useful for testing error handling and recovery.
    """
    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.side_effect = Exception("API Error")
        mock_ticker.return_value = mock_instance
        yield mock_ticker


@pytest.fixture
def ai_trading_config() -> AITradingConfig:
    """Provide a clean AITradingConfig instance for testing.

    Returns:
        AITradingConfig: A fresh AI configuration instance with default values.
    """
    return AITradingConfig()


@pytest.fixture
def data_manager() -> DataManager:
    """Provide a DataManager instance for testing.

    Returns:
        DataManager: A DataManager instance with short cache duration for testing.
    """
    return DataManager(cache_duration=1)  # Very short cache for testing


@pytest.fixture
def ai_trading_strategy() -> TradingStrategy:
    """Provide an AI TradingStrategy instance for testing.

    Returns:
        TradingStrategy: An AI strategy instance with mocked dependencies.
    """
    with patch("core.strategy.get_news_fetcher"), patch(
        "core.strategy.get_ai_analyzer"
    ):
        return TradingStrategy()


@pytest.fixture(autouse=True)
def reset_config_after_test():
    """Auto-reset configuration after each test to ensure test isolation.

    This fixture automatically runs after each test to reset the global
    configuration to default values, preventing test interference.
    """
    yield  # Run the test
    reset_config()  # Clean up after the test


@pytest.fixture
def valid_date_range() -> Dict[str, datetime.date]:
    """Provide a valid date range for testing.

    Returns:
        Dict: Dictionary with start_date and end_date keys.
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=90)
    return {"start_date": start_date, "end_date": end_date}


@pytest.fixture
def invalid_date_range() -> Dict[str, datetime.date]:
    """Provide an invalid date range for testing error cases.

    Returns:
        Dict: Dictionary with start_date after end_date (invalid).
    """
    start_date = datetime.date.today()
    end_date = start_date - datetime.timedelta(days=30)  # End before start
    return {"start_date": start_date, "end_date": end_date}


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions to avoid import errors in tests.

    This fixture mocks Streamlit's UI functions that might be called
    during testing, preventing ImportError and allowing unit testing
    of functions that use Streamlit.
    """
    with patch.dict(
        "sys.modules",
        {
            "streamlit": MagicMock(),
            "streamlit.error": MagicMock(),
            "streamlit.warning": MagicMock(),
            "streamlit.info": MagicMock(),
        },
    ):
        yield


# ===== AI-POWERED FIXTURES (v2.0) =====


@pytest.fixture
def sample_news_articles() -> List[NewsArticle]:
    """Provide sample news articles for AI analysis testing.

    Returns:
        List[NewsArticle]: A collection of realistic news articles with varying sentiment.
    """
    import datetime

    return [
        NewsArticle(
            source="Reuters",
            title="Apple Reports Record Quarterly Earnings, Beats All Expectations",
            description="Apple Inc. posted record quarterly earnings of $3.68 per share, significantly beating analyst expectations of $3.45 per share.",
            url="https://example.com/apple-record-earnings",
            published_at=datetime.datetime(
                2024, 1, 15, 10, 30, 0, tzinfo=datetime.timezone.utc
            ),
            content="Apple Inc. (AAPL) delivered exceptional quarterly results, with revenue growing 15% year-over-year...",
        ),
        NewsArticle(
            source="Bloomberg",
            title="Tech Stocks Face Headwinds Amid Regulatory Concerns",
            description="Major technology companies are facing increased regulatory scrutiny, which could impact future growth prospects.",
            url="https://example.com/tech-regulatory-concerns",
            published_at=datetime.datetime(
                2024, 1, 15, 9, 15, 0, tzinfo=datetime.timezone.utc
            ),
            content="Technology sector giants are navigating a complex regulatory landscape...",
        ),
        NewsArticle(
            source="Financial Times",
            title="Market Volatility Expected to Continue as Fed Policy Remains Uncertain",
            description="Financial markets continue to show signs of volatility as investors await clearer signals from the Federal Reserve.",
            url="https://example.com/market-volatility-fed",
            published_at=datetime.datetime(
                2024, 1, 15, 8, 45, 0, tzinfo=datetime.timezone.utc
            ),
            content="Equity markets have experienced significant swings as uncertainty around monetary policy persists...",
        ),
    ]


@pytest.fixture
def bullish_news_articles() -> List[NewsArticle]:
    """Provide bullish/positive news articles for testing BUY signals.

    Returns:
        List[NewsArticle]: News articles with positive sentiment.
    """
    import datetime

    return [
        NewsArticle(
            source="MarketWatch",
            title="Company Announces Major Breakthrough in AI Technology",
            description="Revolutionary AI advancement positions company as industry leader with significant growth potential.",
            url="https://example.com/ai-breakthrough",
            published_at=datetime.datetime.now(datetime.timezone.utc),
            content="The breakthrough technology is expected to drive substantial revenue growth...",
        ),
        NewsArticle(
            source="CNBC",
            title="Strong Quarterly Results Drive Analyst Upgrades",
            description="Multiple analysts raise price targets following exceptional quarterly performance.",
            url="https://example.com/analyst-upgrades",
            published_at=datetime.datetime.now(datetime.timezone.utc),
            content="Consensus price target increased to $200 from $175 following strong results...",
        ),
    ]


@pytest.fixture
def bearish_news_articles() -> List[NewsArticle]:
    """Provide bearish/negative news articles for testing SELL signals.

    Returns:
        List[NewsArticle]: News articles with negative sentiment.
    """
    import datetime

    return [
        NewsArticle(
            source="Wall Street Journal",
            title="Company Faces Major Lawsuit Over Data Privacy Violations",
            description="Federal lawsuit could result in billions in fines and long-term reputational damage.",
            url="https://example.com/privacy-lawsuit",
            published_at=datetime.datetime.now(datetime.timezone.utc),
            content="The lawsuit alleges systematic privacy violations affecting millions of users...",
        ),
        NewsArticle(
            source="Reuters",
            title="CEO Departure Raises Questions About Company Strategy",
            description="Sudden CEO resignation creates uncertainty about future direction and leadership stability.",
            url="https://example.com/ceo-departure",
            published_at=datetime.datetime.now(datetime.timezone.utc),
            content="The unexpected departure has prompted concerns among investors and analysts...",
        ),
    ]


@pytest.fixture
def neutral_news_articles() -> List[NewsArticle]:
    """Provide neutral news articles for testing HOLD signals.

    Returns:
        List[NewsArticle]: News articles with neutral sentiment.
    """
    import datetime

    return [
        NewsArticle(
            source="Business Wire",
            title="Company Announces Routine Board Meeting Results",
            description="Standard quarterly board meeting concluded with no major announcements or changes.",
            url="https://example.com/board-meeting",
            published_at=datetime.datetime.now(datetime.timezone.utc),
            content="The quarterly board meeting addressed routine operational matters...",
        )
    ]


@pytest.fixture
def mock_ai_analysis_buy() -> AIAnalysisResult:
    """Mock AI analysis result for BUY signal.

    Returns:
        AIAnalysisResult: A bullish AI analysis result.
    """
    import time

    return AIAnalysisResult(
        signal="BUY",
        confidence=0.85,
        rationale="Strong earnings report and positive market sentiment indicate bullish outlook. Multiple analyst upgrades support continued growth potential.",
        model_used="gemini-pro",
        analysis_timestamp=time.time(),
        raw_response='{"signal": "BUY", "confidence": 0.85, "rationale": "Strong earnings report and positive market sentiment..."}',
    )


@pytest.fixture
def mock_ai_analysis_sell() -> AIAnalysisResult:
    """Mock AI analysis result for SELL signal.

    Returns:
        AIAnalysisResult: A bearish AI analysis result.
    """
    import time

    return AIAnalysisResult(
        signal="SELL",
        confidence=0.78,
        rationale="Regulatory concerns and legal challenges pose significant risks to future performance. Recommend reducing exposure.",
        model_used="gemini-pro",
        analysis_timestamp=time.time(),
        raw_response='{"signal": "SELL", "confidence": 0.78, "rationale": "Regulatory concerns and legal challenges..."}',
    )


@pytest.fixture
def mock_ai_analysis_hold() -> AIAnalysisResult:
    """Mock AI analysis result for HOLD signal.

    Returns:
        AIAnalysisResult: A neutral AI analysis result.
    """
    import time

    return AIAnalysisResult(
        signal="HOLD",
        confidence=0.65,
        rationale="Mixed signals from recent news and market conditions. No clear directional bias suggests maintaining current position.",
        model_used="gemini-pro",
        analysis_timestamp=time.time(),
        raw_response='{"signal": "HOLD", "confidence": 0.65, "rationale": "Mixed signals from recent news..."}',
    )


@pytest.fixture
def mock_news_api_success_response() -> Dict[str, Any]:
    """Mock successful NewsAPI response.

    Returns:
        Dict: A mock NewsAPI response with articles.
    """
    return {
        "status": "ok",
        "totalResults": 2,
        "articles": [
            {
                "source": {"id": "reuters", "name": "Reuters"},
                "author": "Jane Reporter",
                "title": "Apple Stock Rises on Strong Earnings Report",
                "description": "Apple Inc. shares climbed after the company reported quarterly results that beat expectations.",
                "url": "https://example.com/apple-earnings-news",
                "urlToImage": "https://example.com/image.jpg",
                "publishedAt": "2024-01-15T10:30:00Z",
                "content": "Apple Inc. reported strong quarterly earnings that exceeded analyst expectations, driving shares higher in after-hours trading...",
            },
            {
                "source": {"id": "bloomberg", "name": "Bloomberg"},
                "author": "John Financial",
                "title": "Technology Sector Shows Resilience Amid Market Volatility",
                "description": "Tech stocks demonstrate strength despite broader market uncertainty.",
                "url": "https://example.com/tech-resilience",
                "urlToImage": "https://example.com/tech-image.jpg",
                "publishedAt": "2024-01-15T09:45:00Z",
                "content": "The technology sector continues to outperform broader market indices despite ongoing economic headwinds...",
            },
        ],
    }


@pytest.fixture
def mock_news_api_error_response() -> Dict[str, Any]:
    """Mock error NewsAPI response.

    Returns:
        Dict: A mock NewsAPI error response.
    """
    return {
        "status": "error",
        "code": "apiKeyInvalid",
        "message": "Your API key is invalid or incorrect. Check your key, or go to https://newsapi.org to create a free API key.",
    }


@pytest.fixture
def mock_gemini_success_response() -> str:
    """Mock successful Gemini API response.

    Returns:
        str: A mock JSON response from Gemini AI.
    """
    return """
    {
        "signal": "BUY",
        "confidence": 0.82,
        "rationale": "The analysis of recent news indicates predominantly positive sentiment. Strong earnings performance and analyst upgrades suggest favorable near-term prospects for the stock."
    }
    """


@pytest.fixture
def mock_gemini_malformed_response() -> str:
    """Mock malformed Gemini API response for error testing.

    Returns:
        str: A malformed JSON response.
    """
    return """
    {
        "signal": "BUY",
        "confidence": 0.82,
        "rationale": "Incomplete JSON response...
    """


@pytest.fixture
def mock_data_with_ai_signals() -> pd.DataFrame:
    """Mock DataFrame with AI signal columns for testing.

    Returns:
        pd.DataFrame: Sample data with AI analysis columns.
    """
    return pd.DataFrame(
        {
            "Date": [datetime.date(2024, 1, i + 1) for i in range(5)],
            "Open": [100.0 + i for i in range(5)],
            "High": [105.0 + i for i in range(5)],
            "Low": [95.0 + i for i in range(5)],
            "Close": [102.0 + i for i in range(5)],
            "Volume": [1000000 + i * 10000 for i in range(5)],
            "Signal": [0, 1, 0, -1, 0],
            "AI_Signal": ["HOLD", "BUY", "HOLD", "SELL", "HOLD"],
            "AI_Confidence": [0.0, 0.85, 0.65, 0.78, 0.55],
            "AI_Rationale": [
                "No analysis performed",
                "Strong positive sentiment from earnings",
                "Mixed market signals",
                "Regulatory concerns outweigh positives",
                "Neutral outlook with limited catalysts",
            ],
        }
    )
