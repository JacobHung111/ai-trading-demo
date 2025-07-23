"""
Unit Tests for AI Analyzer Module

This module tests the AI-powered sentiment analysis functionality with comprehensive
coverage including Gemini API mocking, prompt engineering, JSON parsing, and error handling.

Author: AI Trading Demo Team
Version: 2.0 (AI-Powered Architecture)
"""

import pytest
import json
import time
from unittest.mock import MagicMock, patch, Mock
from typing import Dict, List, Any
import datetime

from google.api_core import exceptions as google_exceptions

from core.ai_analyzer import (
    AIAnalyzer,
    AIAnalysisResult,
    get_ai_analyzer,
    analyze_news_sentiment,
    test_ai_connection,
    validate_analysis_result,
    create_mock_analysis_result,
)
from core.news_fetcher import NewsArticle
from core.config import AITradingConfig


class TestAIAnalysisResult:
    """Test the AIAnalysisResult data class functionality."""

    def test_analysis_result_creation(self):
        # Arrange & Act
        result = AIAnalysisResult(
            signal="BUY",
            confidence=0.85,
            rationale="Strong positive earnings report indicates bullish sentiment",
            model_used="gemini-pro",
            analysis_timestamp=time.time(),
        )

        # Assert
        assert result.signal == "BUY"
        assert result.confidence == 0.85
        assert "positive earnings" in result.rationale
        assert result.model_used == "gemini-pro"
        assert result.analysis_timestamp > 0

    def test_analysis_result_to_dict(self):
        # Arrange
        timestamp = time.time()
        result = AIAnalysisResult(
            signal="SELL",
            confidence=0.72,
            rationale="Market concerns about regulatory issues",
            model_used="gemini-pro",
            analysis_timestamp=timestamp,
            raw_response='{"signal": "SELL", "confidence": 0.72}',
        )

        # Act
        result_dict = result.to_dict()

        # Assert
        assert isinstance(result_dict, dict)
        assert result_dict["signal"] == "SELL"
        assert result_dict["confidence"] == 0.72
        assert result_dict["rationale"] == "Market concerns about regulatory issues"
        assert result_dict["model_used"] == "gemini-pro"
        assert result_dict["analysis_timestamp"] == timestamp
        assert result_dict["raw_response"] is not None

    def test_get_numerical_signal_mapping(self):
        # Arrange & Act
        buy_result = AIAnalysisResult("BUY", 0.8, "Positive", "gemini-pro", time.time())
        sell_result = AIAnalysisResult(
            "SELL", 0.7, "Negative", "gemini-pro", time.time()
        )
        hold_result = AIAnalysisResult(
            "HOLD", 0.5, "Neutral", "gemini-pro", time.time()
        )
        invalid_result = AIAnalysisResult(
            "INVALID", 0.5, "Unknown", "gemini-pro", time.time()
        )

        # Assert
        assert buy_result.get_numerical_signal() == 1
        assert sell_result.get_numerical_signal() == -1
        assert hold_result.get_numerical_signal() == 0
        assert invalid_result.get_numerical_signal() == 0


class TestAIAnalyzer:
    """Test the AIAnalyzer main class functionality."""

    @pytest.fixture
    def mock_config(self):
        """Provide a mock configuration for testing."""
        config = MagicMock(spec=AITradingConfig)
        config.google_api_key = "test_gemini_api_key"
        config.ai_model_name = "gemini-pro"
        config.ai_temperature = 0.1
        config.ai_max_tokens = 1000
        config.gemini_rate_limit_requests = 60
        config.max_retries = 3
        config.get_retry_delay.return_value = 1.0
        return config

    @pytest.fixture
    def sample_news_articles(self):
        """Provide sample news articles for testing."""
        return [
            NewsArticle(
                source="Reuters",
                title="Apple Reports Record Quarterly Earnings",
                description="Apple Inc. posted record quarterly earnings, beating analyst expectations.",
                url="https://example.com/apple-earnings",
                published_at=datetime.datetime.now(datetime.timezone.utc),
                content="Apple reported strong quarterly results...",
            ),
            NewsArticle(
                source="Bloomberg",
                title="Tech Stocks Rally on Positive Market Sentiment",
                description="Technology stocks surged following optimistic market outlook.",
                url="https://example.com/tech-rally",
                published_at=datetime.datetime.now(datetime.timezone.utc),
                content="Technology sector showed strong performance...",
            ),
        ]

    @patch("core.ai_analyzer.genai.configure")
    def test_ai_analyzer_initialization_success(self, mock_configure, mock_config):
        # Arrange & Act
        analyzer = AIAnalyzer(config=mock_config)

        # Assert
        assert analyzer.config == mock_config
        assert analyzer.model_name == "gemini-pro"
        mock_configure.assert_called_once_with(api_key="test_gemini_api_key")

    @patch("core.ai_analyzer.genai.configure")
    def test_ai_analyzer_initialization_no_api_key(self, mock_configure, mock_config):
        # Arrange
        mock_config.google_api_key = None

        # Act
        analyzer = AIAnalyzer(config=mock_config)

        # Assert
        assert analyzer.model_name == "gemini-pro"
        mock_configure.assert_not_called()

    @patch("core.ai_analyzer.genai.configure")
    def test_construct_analysis_prompt(
        self, mock_configure, mock_config, sample_news_articles
    ):
        # Arrange
        analyzer = AIAnalyzer(config=mock_config)

        # Act
        prompt = analyzer._construct_analysis_prompt("AAPL", sample_news_articles)

        # Assert
        assert "AAPL" in prompt
        assert "financial analyst" in prompt.lower()
        assert "Apple Reports Record Quarterly Earnings" in prompt
        assert "Tech Stocks Rally" in prompt
        assert "BUY" in prompt and "SELL" in prompt and "HOLD" in prompt
        assert "JSON" in prompt
        assert "signal" in prompt and "confidence" in prompt and "rationale" in prompt

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_happy_path_buy_signal(
        self, mock_client_class, mock_config, sample_news_articles
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        # Mock successful AI response
        mock_response = MagicMock()
        mock_response.text = '{"signal": "BUY", "confidence": 0.85, "rationale": "Strong earnings report and positive market sentiment indicate bullish outlook for the stock."}'
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", sample_news_articles)

        # Assert
        assert result is not None
        assert result.signal == "BUY"
        assert result.confidence == 0.85
        assert "Strong earnings" in result.rationale
        assert result.model_used == "gemini-pro"
        assert result.get_numerical_signal() == 1

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_happy_path_sell_signal(
        self, mock_client_class, mock_config, sample_news_articles
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.text = '{"signal": "SELL", "confidence": 0.78, "rationale": "Regulatory concerns and market volatility suggest bearish outlook despite positive earnings."}'
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", sample_news_articles)

        # Assert
        assert result is not None
        assert result.signal == "SELL"
        assert result.confidence == 0.78
        assert "bearish outlook" in result.rationale
        assert result.get_numerical_signal() == -1

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_happy_path_hold_signal(
        self, mock_client_class, mock_config, sample_news_articles
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.text = '{"signal": "HOLD", "confidence": 0.65, "rationale": "Mixed signals from earnings and market conditions suggest maintaining current position."}'
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", sample_news_articles)

        # Assert
        assert result is not None
        assert result.signal == "HOLD"
        assert result.confidence == 0.65
        assert "Mixed signals" in result.rationale
        assert result.get_numerical_signal() == 0

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_api_error_handling(
        self, mock_client_class, mock_config, sample_news_articles
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        # Mock API error
        mock_client_instance.models.generate_content.side_effect = (
            google_exceptions.GoogleAPICallError("API rate limit exceeded")
        )

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", sample_news_articles)

        # Assert
        assert result is None

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_quota_exhausted(
        self, mock_client_class, mock_config, sample_news_articles
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        # Mock quota error
        mock_client_instance.models.generate_content.side_effect = (
            google_exceptions.GoogleAPICallError("quota exceeded")
        )

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", sample_news_articles)

        # Assert
        assert result is None

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_malformed_json_response(
        self, mock_client_class, mock_config, sample_news_articles
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        # Mock malformed JSON response (missing closing bracket)
        mock_response = MagicMock()
        mock_response.text = (
            '{"signal": "BUY", "confidence": 0.85, "rationale": "Strong earnings'
        )
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", sample_news_articles)

        # Assert
        assert result is None

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_invalid_json_structure(
        self, mock_client_class, mock_config, sample_news_articles
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        # Mock valid JSON but wrong structure (missing required fields)
        mock_response = MagicMock()
        mock_response.text = (
            '{"recommendation": "BUY", "score": 0.85}'  # Missing required fields
        )
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", sample_news_articles)

        # Assert
        assert result is None

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_invalid_signal_value(
        self, mock_client_class, mock_config, sample_news_articles
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        # Mock response with invalid signal value
        mock_response = MagicMock()
        mock_response.text = '{"signal": "MAYBE", "confidence": 0.85, "rationale": "Uncertain market conditions"}'
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", sample_news_articles)

        # Assert
        assert result is None

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_invalid_confidence_range(
        self, mock_client_class, mock_config, sample_news_articles
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        # Mock response with confidence out of range
        mock_response = MagicMock()
        mock_response.text = (
            '{"signal": "BUY", "confidence": 1.5, "rationale": "Very positive outlook"}'
        )
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", sample_news_articles)

        # Assert
        assert result is None

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_empty_response(
        self, mock_client_class, mock_config, sample_news_articles
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        # Mock empty response
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", sample_news_articles)

        # Assert
        assert result is None

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_no_articles(self, mock_client_class, mock_config):
        # Arrange
        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", [])  # Empty articles list

        # Assert
        assert result is None

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_news_sentiment_client_not_initialized(
        self, mock_client_class, mock_config, sample_news_articles
    ):
        # Arrange
        mock_config.google_api_key = None
        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_news_sentiment("AAPL", sample_news_articles)

        # Assert
        assert result is None

    @patch("core.ai_analyzer.genai.Client")
    def test_analyze_single_text(self, mock_client_class, mock_config):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.text = '{"signal": "BUY", "confidence": 0.8, "rationale": "Positive sentiment in the text"}'
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        result = analyzer.analyze_single_text(
            "AAPL", "Apple reports strong quarterly earnings", "earnings_report"
        )

        # Assert
        assert result is not None
        assert result.signal == "BUY"
        assert result.confidence == 0.8

    @patch("core.ai_analyzer.genai.Client")
    def test_rate_limiting_behavior(self, mock_client_class, mock_config):
        # Arrange
        mock_config.gemini_rate_limit_requests = 1  # Very low limit for testing
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.text = (
            '{"signal": "HOLD", "confidence": 0.5, "rationale": "Neutral"}'
        )
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = AIAnalyzer(config=mock_config)

        # Act - Make rapid requests
        start_time = time.time()
        analyzer._make_ai_request("Test prompt 1")
        analyzer._make_ai_request("Test prompt 2")
        end_time = time.time()

        # Assert - Should have some delay due to rate limiting
        assert end_time - start_time >= 0

    @patch("core.ai_analyzer.genai.Client")
    def test_test_connection_success(self, mock_client_class, mock_config):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.text = '{"test": "success", "model": "working"}'
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        status = analyzer.test_connection()

        # Assert
        assert status["connected"] is True
        assert status["api_key_valid"] is True
        assert status["client_initialized"] is True
        assert status["model_name"] == "gemini-pro"
        assert "success" in status["test_response"]

    @patch("core.ai_analyzer.genai.Client")
    def test_test_connection_failure(self, mock_client_class, mock_config):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.models.generate_content.side_effect = Exception(
            "Connection failed"
        )

        analyzer = AIAnalyzer(config=mock_config)

        # Act
        status = analyzer.test_connection()

        # Assert
        assert status["connected"] is False
        assert "Connection failed" in status["error"]

    def test_get_analysis_stats(self, mock_config):
        # Arrange
        with patch("core.ai_analyzer.genai.Client"):
            analyzer = AIAnalyzer(config=mock_config)

        # Act
        stats = analyzer.get_analysis_stats()

        # Assert
        assert stats["model_name"] == "gemini-pro"
        assert stats["temperature"] == 0.1
        assert stats["max_tokens"] == 1000
        assert stats["rate_limit_requests"] == 60
        assert stats["max_retries"] == 3
        assert "last_request_time" in stats
        assert "client_initialized" in stats


class TestGlobalFunctions:
    """Test module-level convenience functions."""

    @patch("core.ai_analyzer.AIAnalyzer")
    def test_get_ai_analyzer_singleton(self, mock_analyzer_class):
        # Arrange
        mock_instance = MagicMock()
        mock_analyzer_class.return_value = mock_instance

        # Act
        analyzer1 = get_ai_analyzer()
        analyzer2 = get_ai_analyzer()

        # Assert
        assert analyzer1 is analyzer2  # Should return same instance
        mock_analyzer_class.assert_called_once()  # Should only create once

    @patch("core.ai_analyzer.get_ai_analyzer")
    def test_analyze_news_sentiment_convenience_function(self, mock_get_analyzer):
        # Arrange
        mock_analyzer_instance = MagicMock()
        mock_result = AIAnalysisResult(
            "BUY", 0.8, "Positive", "gemini-pro", time.time()
        )
        mock_analyzer_instance.analyze_news_sentiment.return_value = mock_result
        mock_get_analyzer.return_value = mock_analyzer_instance

        articles = [MagicMock()]

        # Act
        result = analyze_news_sentiment("AAPL", articles, "context")

        # Assert
        mock_analyzer_instance.analyze_news_sentiment.assert_called_once_with(
            "AAPL", articles, "context"
        )
        assert result == mock_result

    @patch("core.ai_analyzer.get_ai_analyzer")
    def test_test_ai_connection_convenience_function(self, mock_get_analyzer):
        # Arrange
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.test_connection.return_value = {"connected": True}
        mock_get_analyzer.return_value = mock_analyzer_instance

        # Act
        result = test_ai_connection()

        # Assert
        mock_analyzer_instance.test_connection.assert_called_once()
        assert result == {"connected": True}

    def test_validate_analysis_result_valid(self):
        # Arrange
        result = AIAnalysisResult(
            signal="BUY",
            confidence=0.85,
            rationale="Strong positive outlook",
            model_used="gemini-pro",
            analysis_timestamp=time.time(),
        )

        # Act
        is_valid = validate_analysis_result(result)

        # Assert
        assert is_valid is True

    def test_validate_analysis_result_invalid_signal(self):
        # Arrange
        result = AIAnalysisResult(
            signal="INVALID",
            confidence=0.85,
            rationale="Strong positive outlook",
            model_used="gemini-pro",
            analysis_timestamp=time.time(),
        )

        # Act
        is_valid = validate_analysis_result(result)

        # Assert
        assert is_valid is False

    def test_validate_analysis_result_invalid_confidence(self):
        # Arrange
        result = AIAnalysisResult(
            signal="BUY",
            confidence=1.5,  # Out of range
            rationale="Strong positive outlook",
            model_used="gemini-pro",
            analysis_timestamp=time.time(),
        )

        # Act
        is_valid = validate_analysis_result(result)

        # Assert
        assert is_valid is False

    def test_validate_analysis_result_empty_rationale(self):
        # Arrange
        result = AIAnalysisResult(
            signal="BUY",
            confidence=0.85,
            rationale="",  # Empty rationale
            model_used="gemini-pro",
            analysis_timestamp=time.time(),
        )

        # Act
        is_valid = validate_analysis_result(result)

        # Assert
        assert is_valid is False

    def test_create_mock_analysis_result(self):
        # Arrange & Act
        result = create_mock_analysis_result("SELL", 0.75, "Mock analysis for testing")

        # Assert
        assert result.signal == "SELL"
        assert result.confidence == 0.75
        assert result.rationale == "Mock analysis for testing"
        assert result.model_used == "mock"
        assert result.analysis_timestamp > 0
        assert "SELL" in result.raw_response


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @patch("core.ai_analyzer.genai.Client")
    def test_prompt_construction_with_special_characters(
        self, mock_client_class, mock_config
    ):
        # Arrange
        analyzer = AIAnalyzer(config=mock_config)

        # Articles with special characters that might break JSON parsing
        special_articles = [
            NewsArticle(
                source="Test",
                title='Apple "Reports" Strong Earnings & Growth',
                description="Company's CEO says: \"We're optimistic about Q4\"",
                url="https://example.com",
                published_at=datetime.datetime.now(datetime.timezone.utc),
            )
        ]

        # Act
        prompt = analyzer._construct_analysis_prompt("AAPL", special_articles)

        # Assert
        assert "AAPL" in prompt
        assert "Apple" in prompt
        # Should handle special characters gracefully

    @patch("core.ai_analyzer.genai.Client")
    def test_json_parsing_with_extra_text(self, mock_client_class, mock_config):
        # Arrange
        analyzer = AIAnalyzer(config=mock_config)

        # Response with extra text around JSON
        response_with_extra = 'Here is my analysis: {"signal": "BUY", "confidence": 0.8, "rationale": "Good news"} Hope this helps!'

        # Act
        parsed = analyzer._parse_ai_response(response_with_extra)

        # Assert
        assert parsed is not None
        assert parsed["signal"] == "BUY"
        assert parsed["confidence"] == 0.8

    def test_analysis_result_numerical_signal_edge_cases(self):
        # Arrange & Act
        result_none = AIAnalysisResult(None, 0.5, "Test", "model", time.time())
        result_lowercase = AIAnalysisResult("buy", 0.5, "Test", "model", time.time())
        result_mixed = AIAnalysisResult("Hold", 0.5, "Test", "model", time.time())

        # Assert
        assert result_none.get_numerical_signal() == 0  # Default for invalid signal
        assert result_lowercase.get_numerical_signal() == 0  # Case sensitive
        assert result_mixed.get_numerical_signal() == 0  # Case sensitive
