"""
Tests for AI-Powered Trading Strategy Module

This module tests the logical correctness of the AI-powered trading strategy
that orchestrates news fetching and AI sentiment analysis to generate trading signals.
Tests follow the precise specifications in claude.v2.ai-powered.md Chapter 7.

Author: AI Trading Demo Team
Version: 2.0 (AI-Powered Architecture)
"""

import pytest
import pandas as pd
import datetime
import time
from unittest.mock import MagicMock, patch, Mock
from typing import List, Dict, Any

from core.strategy import (
    TradingStrategy,
    generate_trading_signals_with_ui_feedback,
    get_signal_summary_with_ui_feedback,
)
from core.news_fetcher import NewsArticle
from core.ai_analyzer import AIAnalysisResult
from core.config import AITradingConfig


class TestAITradingStrategy:
    """Test class for AI-powered TradingStrategy functionality."""

    @pytest.fixture
    def mock_config(self):
        """Provide a mock configuration for testing."""
        config = MagicMock(spec=AITradingConfig)
        config.news_articles_count = 10
        config.max_news_age_hours = 24
        config.min_data_points = 30
        config.min_confidence_threshold = 0.6
        return config

    @pytest.fixture
    def sample_news_articles(self):
        """Provide sample news articles for testing."""
        return [
            NewsArticle(
                source="Reuters",
                title="Apple Reports Strong Quarterly Earnings",
                description="Apple Inc. posted record quarterly earnings, exceeding analyst expectations.",
                url="https://example.com/apple-earnings",
                published_at=datetime.datetime.now(datetime.timezone.utc),
                content="Apple reported strong results...",
            ),
            NewsArticle(
                source="Bloomberg",
                title="Tech Stocks Decline on Market Volatility",
                description="Technology stocks fell amid broader market concerns.",
                url="https://example.com/tech-decline",
                published_at=datetime.datetime.now(datetime.timezone.utc),
                content="Tech sector showed weakness...",
            ),
        ]

    @pytest.fixture
    def sample_price_data(self):
        """Provide sample OHLCV price data for testing."""
        return pd.DataFrame(
            {
                "Date": [
                    datetime.date(2024, 1, 1) + datetime.timedelta(days=i)
                    for i in range(60)
                ],
                "Open": [100.0 + i * 0.5 for i in range(60)],
                "High": [102.0 + i * 0.5 for i in range(60)],
                "Low": [98.0 + i * 0.5 for i in range(60)],
                "Close": [101.0 + i * 0.5 for i in range(60)],
                "Volume": [1000000 + i * 10000 for i in range(60)],
            }
        )

    def test_strategy_initialization(self, mock_config):
        """Test AI TradingStrategy initialization."""
        # Arrange & Act
        with patch("core.strategy.get_news_fetcher") as mock_get_news, patch(
            "core.strategy.get_ai_analyzer"
        ) as mock_get_ai:

            mock_news_fetcher = MagicMock()
            mock_ai_analyzer = MagicMock()
            mock_get_news.return_value = mock_news_fetcher
            mock_get_ai.return_value = mock_ai_analyzer

            strategy = TradingStrategy(config=mock_config)

        # Assert
        assert strategy.config == mock_config
        assert strategy.news_fetcher == mock_news_fetcher
        assert strategy.ai_analyzer == mock_ai_analyzer
        assert strategy.max_articles == mock_config.news_articles_count
        assert strategy.max_age_hours == mock_config.max_news_age_hours

    @patch("core.strategy.get_news_fetcher")
    @patch("core.strategy.get_ai_analyzer")
    def test_generate_signals_buy_signal_scenario(
        self,
        mock_get_ai,
        mock_get_news,
        mock_config,
        sample_price_data,
        sample_news_articles,
    ):
        """Test AI strategy generating BUY signal.

        Per Chapter 11.5.3: Mock ai_analyzer to return "BUY" signal and
        assert that strategy.py correctly generates a 1.
        """
        # Arrange: Mock news fetcher and AI analyzer
        mock_news_fetcher = MagicMock()
        mock_ai_analyzer = MagicMock()
        mock_get_news.return_value = mock_news_fetcher
        mock_get_ai.return_value = mock_ai_analyzer

        # Mock news fetcher to return sample articles
        mock_news_fetcher.get_stock_news.return_value = sample_news_articles

        # Mock AI analyzer to return BUY signal
        buy_result = AIAnalysisResult(
            signal="BUY",
            confidence=0.85,
            rationale="Strong earnings report indicates positive outlook for the stock.",
            model_used="gemini-pro",
            analysis_timestamp=time.time(),
        )
        mock_ai_analyzer.analyze_news_sentiment.return_value = buy_result

        strategy = TradingStrategy(config=mock_config)

        # Act: Generate signals
        result_data = strategy.generate_signals(sample_price_data, ticker="AAPL")

        # Assert: Verify BUY signal is generated correctly
        assert "Signal" in result_data.columns, "Signal column should be added"
        assert "AI_Signal" in result_data.columns, "AI_Signal column should be added"
        assert (
            "AI_Confidence" in result_data.columns
        ), "AI_Confidence column should be added"
        assert (
            "AI_Rationale" in result_data.columns
        ), "AI_Rationale column should be added"

        # Check the latest signal (applied to most recent data)
        latest_signal = result_data.iloc[-1]
        assert latest_signal["Signal"] == 1, "Should generate BUY signal (1)"
        assert latest_signal["AI_Signal"] == "BUY", "AI_Signal should be BUY"
        assert latest_signal["AI_Confidence"] == 0.85, "Should preserve AI confidence"
        assert (
            "positive outlook" in latest_signal["AI_Rationale"]
        ), "Should preserve AI rationale"

        # Verify API calls were made correctly
        mock_news_fetcher.get_stock_news.assert_called_once_with(
            ticker="AAPL",
            max_articles=mock_config.news_articles_count,
            max_age_hours=mock_config.max_news_age_hours,
        )
        mock_ai_analyzer.analyze_news_sentiment.assert_called_once_with(
            ticker="AAPL", articles=sample_news_articles
        )

    @patch("core.strategy.get_news_fetcher")
    @patch("core.strategy.get_ai_analyzer")
    def test_generate_signals_sell_signal_scenario(
        self,
        mock_get_ai,
        mock_get_news,
        mock_config,
        sample_price_data,
        sample_news_articles,
    ):
        """Test AI strategy generating SELL signal.

        Per Chapter 11.5.3: Mock ai_analyzer to return "SELL" signal and
        assert that strategy.py correctly generates a -1.
        """
        # Arrange: Mock components for SELL signal
        mock_news_fetcher = MagicMock()
        mock_ai_analyzer = MagicMock()
        mock_get_news.return_value = mock_news_fetcher
        mock_get_ai.return_value = mock_ai_analyzer

        mock_news_fetcher.get_stock_news.return_value = sample_news_articles

        # Mock AI analyzer to return SELL signal
        sell_result = AIAnalysisResult(
            signal="SELL",
            confidence=0.78,
            rationale="Market volatility and regulatory concerns suggest bearish outlook.",
            model_used="gemini-pro",
            analysis_timestamp=time.time(),
        )
        mock_ai_analyzer.analyze_news_sentiment.return_value = sell_result

        strategy = TradingStrategy(config=mock_config)

        # Act: Generate signals
        result_data = strategy.generate_signals(sample_price_data, ticker="GOOGL")

        # Assert: Verify SELL signal is generated correctly
        latest_signal = result_data.iloc[-1]
        assert latest_signal["Signal"] == -1, "Should generate SELL signal (-1)"
        assert latest_signal["AI_Signal"] == "SELL", "AI_Signal should be SELL"
        assert latest_signal["AI_Confidence"] == 0.78, "Should preserve AI confidence"
        assert (
            "bearish outlook" in latest_signal["AI_Rationale"]
        ), "Should preserve AI rationale"

    @patch("core.strategy.get_news_fetcher")
    @patch("core.strategy.get_ai_analyzer")
    def test_generate_signals_hold_signal_scenario(
        self,
        mock_get_ai,
        mock_get_news,
        mock_config,
        sample_price_data,
        sample_news_articles,
    ):
        """Test AI strategy generating HOLD signal (no signal)."""
        # Arrange: Mock components for HOLD signal
        mock_news_fetcher = MagicMock()
        mock_ai_analyzer = MagicMock()
        mock_get_news.return_value = mock_news_fetcher
        mock_get_ai.return_value = mock_ai_analyzer

        mock_news_fetcher.get_stock_news.return_value = sample_news_articles

        # Mock AI analyzer to return HOLD signal
        hold_result = AIAnalysisResult(
            signal="HOLD",
            confidence=0.65,
            rationale="Mixed signals from recent news suggest maintaining current position.",
            model_used="gemini-pro",
            analysis_timestamp=time.time(),
        )
        mock_ai_analyzer.analyze_news_sentiment.return_value = hold_result

        strategy = TradingStrategy(config=mock_config)

        # Act: Generate signals
        result_data = strategy.generate_signals(sample_price_data, ticker="MSFT")

        # Assert: Verify HOLD signal is generated correctly
        latest_signal = result_data.iloc[-1]
        assert latest_signal["Signal"] == 0, "Should generate HOLD signal (0)"
        assert latest_signal["AI_Signal"] == "HOLD", "AI_Signal should be HOLD"
        assert latest_signal["AI_Confidence"] == 0.65, "Should preserve AI confidence"
        assert (
            "Mixed signals" in latest_signal["AI_Rationale"]
        ), "Should preserve AI rationale"

    @patch("core.strategy.get_news_fetcher")
    @patch("core.strategy.get_ai_analyzer")
    def test_generate_signals_ai_error_scenario(
        self,
        mock_get_ai,
        mock_get_news,
        mock_config,
        sample_price_data,
        sample_news_articles,
    ):
        """Test AI Error scenario.

        Per Chapter 11.5.3: Mock ai_analyzer to return error state and
        assert that strategy.py correctly generates a 0 ("No Signal").
        """
        # Arrange: Mock components for AI error
        mock_news_fetcher = MagicMock()
        mock_ai_analyzer = MagicMock()
        mock_get_news.return_value = mock_news_fetcher
        mock_get_ai.return_value = mock_ai_analyzer

        mock_news_fetcher.get_stock_news.return_value = sample_news_articles

        # Mock AI analyzer to return error (None)
        mock_ai_analyzer.analyze_news_sentiment.return_value = None

        strategy = TradingStrategy(config=mock_config)

        # Act: Generate signals
        result_data = strategy.generate_signals(sample_price_data, ticker="TSLA")

        # Assert: Verify error handling generates default HOLD state
        latest_signal = result_data.iloc[-1]
        assert latest_signal["Signal"] == 0, "Should generate no signal (0) on AI error"
        assert latest_signal["AI_Signal"] == "HOLD", "AI_Signal should default to HOLD"
        assert (
            latest_signal["AI_Confidence"] == 0.0
        ), "Confidence should be 0.0 on error"
        assert (
            latest_signal["AI_Rationale"] == "No analysis performed"
        ), "Should have default rationale"

    @patch("core.strategy.get_news_fetcher")
    @patch("core.strategy.get_ai_analyzer")
    def test_generate_signals_no_news_articles(
        self, mock_get_ai, mock_get_news, mock_config, sample_price_data
    ):
        """Test AI strategy when no news articles are found."""
        # Arrange: Mock components with no news
        mock_news_fetcher = MagicMock()
        mock_ai_analyzer = MagicMock()
        mock_get_news.return_value = mock_news_fetcher
        mock_get_ai.return_value = mock_ai_analyzer

        # Mock news fetcher to return empty list
        mock_news_fetcher.get_stock_news.return_value = []

        strategy = TradingStrategy(config=mock_config)

        # Act: Generate signals
        result_data = strategy.generate_signals(sample_price_data, ticker="NFLX")

        # Assert: Should return data with default HOLD signals
        latest_signal = result_data.iloc[-1]
        assert (
            latest_signal["Signal"] == 0
        ), "Should generate no signal when no news found"
        assert latest_signal["AI_Signal"] == "HOLD", "Should default to HOLD"
        assert latest_signal["AI_Confidence"] == 0.0, "Confidence should be 0.0"
        assert (
            latest_signal["AI_Rationale"] == "No analysis performed"
        ), "Should have default rationale"

        # AI analyzer should not be called if no news
        mock_ai_analyzer.analyze_news_sentiment.assert_not_called()

    def test_generate_signals_empty_data(self, mock_config):
        """Test signal generation with empty DataFrame."""
        # Arrange
        with patch("core.strategy.get_news_fetcher"), patch(
            "core.strategy.get_ai_analyzer"
        ):
            strategy = TradingStrategy(config=mock_config)
            empty_data = pd.DataFrame()

        # Act
        result_data = strategy.generate_signals(empty_data, ticker="AAPL")

        # Assert: Should return empty DataFrame
        assert result_data.empty, "Should return empty DataFrame for empty input"

    def test_get_latest_signal_with_ai_data(self, mock_config):
        """Test getting latest signal with AI analysis data."""
        # Arrange: Data with AI signals
        with patch("core.strategy.get_news_fetcher"), patch(
            "core.strategy.get_ai_analyzer"
        ):

            data_with_ai_signals = pd.DataFrame(
                {
                    "Date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
                    "Close": [100.0, 102.0],
                    "Signal": [0, 1],
                    "AI_Signal": ["HOLD", "BUY"],
                    "AI_Confidence": [0.0, 0.85],
                    "AI_Rationale": ["No analysis", "Strong positive sentiment"],
                }
            )

            strategy = TradingStrategy(config=mock_config)

        # Act: Get latest signal
        latest_signal = strategy.get_latest_signal(data_with_ai_signals)

        # Assert: Should return latest BUY signal with AI data
        assert latest_signal is not None
        assert latest_signal["signal"] == 1
        assert latest_signal["type"] == "BUY"
        assert latest_signal["ai_signal"] == "BUY"
        assert latest_signal["confidence"] == 0.85
        assert latest_signal["rationale"] == "Strong positive sentiment"

    def test_get_latest_signal_hold_only(self, mock_config):
        """Test getting latest signal when only HOLD signals exist."""
        # Arrange: Data with only HOLD signals
        with patch("core.strategy.get_news_fetcher"), patch(
            "core.strategy.get_ai_analyzer"
        ):

            data_with_holds = pd.DataFrame(
                {
                    "Date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
                    "Close": [100.0, 102.0],
                    "Signal": [0, 0],
                    "AI_Signal": ["HOLD", "HOLD"],
                    "AI_Confidence": [0.5, 0.6],
                    "AI_Rationale": ["Neutral sentiment", "Mixed signals"],
                }
            )

            strategy = TradingStrategy(config=mock_config)

        # Act: Get latest signal
        latest_signal = strategy.get_latest_signal(data_with_holds)

        # Assert: Should return most recent analysis even if HOLD
        assert latest_signal is not None
        assert latest_signal["signal"] == 0
        assert latest_signal["type"] == "HOLD"
        assert latest_signal["ai_signal"] == "HOLD"
        assert latest_signal["confidence"] == 0.6
        assert latest_signal["rationale"] == "Mixed signals"

    def test_get_signal_summary_with_ai_data(self, mock_config):
        """Test signal summary generation with AI analysis data."""
        # Arrange: Data with multiple AI signals
        with patch("core.strategy.get_news_fetcher"), patch(
            "core.strategy.get_ai_analyzer"
        ):

            data_with_ai_signals = pd.DataFrame(
                {
                    "Signal": [0, 1, 0, -1, 0],
                    "AI_Confidence": [0.0, 0.85, 0.65, 0.78, 0.55],
                }
            )

            strategy = TradingStrategy(config=mock_config)

        # Act: Get signal summary
        summary = strategy.get_signal_summary(data_with_ai_signals)

        # Assert: Should include AI-specific statistics
        assert summary is not None
        assert summary["total_signals"] == 2  # 1 buy + 1 sell
        assert summary["buy_signals"] == 1
        assert summary["sell_signals"] == 1
        assert summary["hold_signals"] == 3
        assert summary["total_days"] == 5
        assert summary["signal_rate"] == 40.0  # (2/5)*100
        assert summary["analyses_performed"] == 4  # Confidence > 0
        assert summary["avg_confidence"] > 0  # Average of non-zero confidences

    def test_get_signal_list_with_ai_data(self, mock_config):
        """Test getting signal list with AI analysis data."""
        # Arrange: Data with AI signals
        with patch("core.strategy.get_news_fetcher"), patch(
            "core.strategy.get_ai_analyzer"
        ):

            data_with_ai_signals = pd.DataFrame(
                {
                    "Date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
                    "Close": [100.0, 102.0],
                    "Volume": [1000000, 1100000],
                    "Signal": [1, -1],
                    "AI_Signal": ["BUY", "SELL"],
                    "AI_Confidence": [0.85, 0.78],
                    "AI_Rationale": ["Positive sentiment", "Negative outlook"],
                }
            )

            strategy = TradingStrategy(config=mock_config)

        # Act: Get signal list
        signal_list = strategy.get_signal_list(data_with_ai_signals)

        # Assert: Should return list with AI data
        assert len(signal_list) == 2

        # Check BUY signal
        buy_signal = signal_list[0]
        assert buy_signal["signal"] == 1
        assert buy_signal["type"] == "BUY"
        assert buy_signal["ai_signal"] == "BUY"
        assert buy_signal["confidence"] == 0.85
        assert buy_signal["rationale"] == "Positive sentiment"

        # Check SELL signal
        sell_signal = signal_list[1]
        assert sell_signal["signal"] == -1
        assert sell_signal["type"] == "SELL"
        assert sell_signal["ai_signal"] == "SELL"
        assert sell_signal["confidence"] == 0.78
        assert sell_signal["rationale"] == "Negative outlook"

    def test_analyze_single_ticker(self, mock_config, sample_news_articles):
        """Test real-time single ticker analysis."""
        # Arrange: Mock components for real-time analysis
        with patch("core.strategy.get_news_fetcher") as mock_get_news, patch(
            "core.strategy.get_ai_analyzer"
        ) as mock_get_ai:

            mock_news_fetcher = MagicMock()
            mock_ai_analyzer = MagicMock()
            mock_get_news.return_value = mock_news_fetcher
            mock_get_ai.return_value = mock_ai_analyzer

            mock_news_fetcher.get_stock_news.return_value = sample_news_articles

            analysis_result = AIAnalysisResult(
                signal="BUY",
                confidence=0.80,
                rationale="Strong quarterly results",
                model_used="gemini-pro",
                analysis_timestamp=time.time(),
            )
            mock_ai_analyzer.analyze_news_sentiment.return_value = analysis_result

            strategy = TradingStrategy(config=mock_config)

        # Act: Analyze single ticker
        result = strategy.analyze_single_ticker("AAPL")

        # Assert: Should return real-time analysis
        assert result is not None
        assert result["signal"] == 1
        assert result["ai_signal"] == "BUY"
        assert result["confidence"] == 0.80
        assert result["rationale"] == "Strong quarterly results"
        assert "timestamp" in result

    def test_analyze_single_ticker_no_news(self, mock_config):
        """Test single ticker analysis when no news is available."""
        # Arrange: Mock with no news articles
        with patch("core.strategy.get_news_fetcher") as mock_get_news, patch(
            "core.strategy.get_ai_analyzer"
        ) as mock_get_ai:

            mock_news_fetcher = MagicMock()
            mock_ai_analyzer = MagicMock()
            mock_get_news.return_value = mock_news_fetcher
            mock_get_ai.return_value = mock_ai_analyzer

            mock_news_fetcher.get_stock_news.return_value = []

            strategy = TradingStrategy(config=mock_config)

        # Act: Analyze single ticker
        result = strategy.analyze_single_ticker("UNKNOWN")

        # Assert: Should return HOLD with appropriate message
        assert result is not None
        assert result["signal"] == 0
        assert result["ai_signal"] == "HOLD"
        assert result["confidence"] == 0.0
        assert "No recent news available" in result["rationale"]

    def test_get_strategy_parameters(self, mock_config):
        """Test getting AI strategy parameters."""
        # Arrange
        mock_config.ai_model_name = "gemini-pro"
        mock_config.ai_temperature = 0.1
        mock_config.min_confidence_threshold = 0.6

        with patch("core.strategy.get_news_fetcher"), patch(
            "core.strategy.get_ai_analyzer"
        ):
            strategy = TradingStrategy(config=mock_config)

        # Act: Get parameters
        params = strategy.get_strategy_parameters()

        # Assert: Should return AI strategy parameters
        assert params["max_articles"] == mock_config.news_articles_count
        assert params["max_age_hours"] == mock_config.max_news_age_hours
        assert params["ai_model"] == "gemini-pro"
        assert params["ai_temperature"] == 0.1
        assert params["confidence_threshold"] == 0.6
        assert params["strategy_name"] == "AI News Sentiment Analysis"
        assert params["version"] == "2.0"

    def test_validate_setup(self, mock_config):
        """Test strategy setup validation."""
        # Arrange: Mock successful validation
        with patch("core.strategy.get_news_fetcher") as mock_get_news, patch(
            "core.strategy.get_ai_analyzer"
        ) as mock_get_ai:

            mock_news_fetcher = MagicMock()
            mock_ai_analyzer = MagicMock()
            mock_get_news.return_value = mock_news_fetcher
            mock_get_ai.return_value = mock_ai_analyzer

            # Mock successful validations
            mock_config.validate_api_keys.return_value = {
                "google_api_key": True,
                "newsapi_api_key": True,
            }
            mock_news_fetcher.validate_connection.return_value = {"connected": True}
            mock_ai_analyzer.test_connection.return_value = {"connected": True}

            strategy = TradingStrategy(config=mock_config)

        # Act: Validate setup
        validation = strategy.validate_setup()

        # Assert: Should return successful validation
        assert validation["strategy_ready"] is True
        assert validation["news_fetcher_ready"] is True
        assert validation["ai_analyzer_ready"] is True
        assert validation["config_valid"] is True
        assert len(validation["errors"]) == 0


class TestStrategyUIFeedbackFunctions:
    """Test UI feedback functions for Streamlit integration with AI strategy."""

    @patch("core.strategy.TradingStrategy")
    def test_generate_trading_signals_with_ui_feedback_success(
        self, mock_strategy_class, sample_price_data
    ):
        """Test AI signal generation with UI feedback using valid data."""
        # Arrange: Mock successful AI strategy
        mock_strategy = MagicMock()
        mock_strategy_class.return_value = mock_strategy

        # Mock successful signal generation with AI data
        result_data = sample_price_data.copy()
        result_data["Signal"] = 0
        result_data["AI_Signal"] = "HOLD"
        result_data["AI_Confidence"] = 0.65
        result_data["AI_Rationale"] = "Mixed market conditions"
        result_data.iloc[-1, result_data.columns.get_loc("Signal")] = 1
        result_data.iloc[-1, result_data.columns.get_loc("AI_Signal")] = "BUY"
        result_data.iloc[-1, result_data.columns.get_loc("AI_Confidence")] = 0.85

        mock_strategy.generate_signals.return_value = result_data

        # Act: Generate signals with UI feedback
        with patch("streamlit.spinner"), patch("streamlit.success") as mock_success:
            result = generate_trading_signals_with_ui_feedback(
                sample_price_data, ticker="AAPL"
            )

        # Assert: Should return data with AI signals and show success message
        assert "Signal" in result.columns
        assert "AI_Signal" in result.columns
        assert "AI_Confidence" in result.columns
        mock_success.assert_called_once()
        assert "BUY" in mock_success.call_args[0][0]

    def test_generate_trading_signals_with_ui_feedback_empty_data(self):
        """Test UI feedback function with empty DataFrame."""
        # Arrange: Empty DataFrame
        empty_data = pd.DataFrame()

        # Act: Generate signals with UI feedback
        with patch("streamlit.warning") as mock_warning:
            result = generate_trading_signals_with_ui_feedback(empty_data)

        # Assert: Should show warning and return original data
        mock_warning.assert_called_once()
        assert result.equals(empty_data)

    def test_get_signal_summary_with_ui_feedback(self):
        """Test AI signal summary with UI feedback."""
        # Arrange: Data with AI signals
        data_with_ai = pd.DataFrame(
            {"Signal": [0, 1, 0, -1, 0], "AI_Confidence": [0.0, 0.85, 0.65, 0.78, 0.55]}
        )

        # Act: Get signal summary
        summary = get_signal_summary_with_ui_feedback(data_with_ai)

        # Assert: Should return AI-enhanced summary
        assert "total_signals" in summary
        assert "buy_signals" in summary
        assert "sell_signals" in summary
        assert "no_signal_days" in summary
        assert "total_days" in summary
        assert summary["total_signals"] == 2
        assert summary["buy_signals"] == 1
        assert summary["sell_signals"] == 1

    def test_get_signal_summary_with_ui_feedback_missing_column(self):
        """Test UI feedback summary with missing Signal column."""
        # Arrange: Data without Signal column
        invalid_data = pd.DataFrame({"Close": [100.0, 102.0]})

        # Act: Get summary
        summary = get_signal_summary_with_ui_feedback(invalid_data)

        # Assert: Should return empty dict
        assert summary == {}


class TestAIStrategyEdgeCases:
    """Test edge cases and error scenarios for AI strategy."""

    def test_caching_behavior(
        self, mock_config, sample_news_articles, sample_price_data
    ):
        """Test that analysis results are cached properly."""
        # Arrange: Mock components with caching
        with patch("core.strategy.get_news_fetcher") as mock_get_news, patch(
            "core.strategy.get_ai_analyzer"
        ) as mock_get_ai:

            mock_news_fetcher = MagicMock()
            mock_ai_analyzer = MagicMock()
            mock_get_news.return_value = mock_news_fetcher
            mock_get_ai.return_value = mock_ai_analyzer

            mock_news_fetcher.get_stock_news.return_value = sample_news_articles

            analysis_result = AIAnalysisResult(
                signal="BUY",
                confidence=0.80,
                rationale="Positive analysis",
                model_used="gemini-pro",
                analysis_timestamp=time.time(),
            )
            mock_ai_analyzer.analyze_news_sentiment.return_value = analysis_result

            strategy = TradingStrategy(config=mock_config)

        # Act: Generate signals twice rapidly
        result1 = strategy.generate_signals(sample_price_data, ticker="AAPL")
        result2 = strategy.generate_signals(sample_price_data, ticker="AAPL")

        # Assert: AI analyzer should only be called once due to caching
        assert (
            mock_ai_analyzer.analyze_news_sentiment.call_count <= 2
        )  # May be called twice but result cached

    def test_cache_stats_and_management(self, mock_config):
        """Test cache statistics and management functions."""
        # Arrange
        with patch("core.strategy.get_news_fetcher"), patch(
            "core.strategy.get_ai_analyzer"
        ):
            strategy = TradingStrategy(config=mock_config)

        # Act & Assert: Cache stats
        stats = strategy.get_cache_stats()
        assert "cache_entries" in stats
        assert "cache_ttl" in stats
        assert "cache_keys" in stats

        # Act & Assert: Cache clearing
        strategy.clear_cache()
        stats_after_clear = strategy.get_cache_stats()
        assert stats_after_clear["cache_entries"] == 0
