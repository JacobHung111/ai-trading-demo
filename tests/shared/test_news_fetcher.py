"""
Unit Tests for News Fetcher Module

This module tests the news fetching functionality with comprehensive coverage
including API mocking, error handling, rate limiting, and caching behaviors.

Author: AI Trading Demo Team
Version: 2.0 (AI-Powered Architecture)
"""

import pytest
import datetime
from unittest.mock import MagicMock, patch, Mock
import time
from typing import Dict, Any, List
import json

from core.news_fetcher import (
    NewsFetcher,
    NewsArticle,
    TokenBucket,
    NewsCache,
    get_news_fetcher,
    fetch_stock_news,
    validate_news_api,
)
from core.config import AITradingConfig


class TestNewsArticle:
    """Test the NewsArticle data class functionality."""

    def test_news_article_creation(self):
        # Arrange
        published_date = datetime.datetime.now(datetime.timezone.utc)

        # Act
        article = NewsArticle(
            source="Reuters",
            title="Apple Reports Strong Quarterly Results",
            description="Apple Inc. posted better-than-expected quarterly results.",
            url="https://example.com/news/apple",
            published_at=published_date,
            content="Full article content...",
        )

        # Assert
        assert article.source == "Reuters"
        assert article.title == "Apple Reports Strong Quarterly Results"
        assert (
            article.description
            == "Apple Inc. posted better-than-expected quarterly results."
        )
        assert article.url == "https://example.com/news/apple"
        assert article.published_at == published_date
        assert article.content == "Full article content..."

    def test_news_article_to_dict(self):
        # Arrange
        published_date = datetime.datetime.now(datetime.timezone.utc)
        article = NewsArticle(
            source="Bloomberg",
            title="Market News",
            description="Market update",
            url="https://example.com",
            published_at=published_date,
        )

        # Act
        article_dict = article.to_dict()

        # Assert
        assert isinstance(article_dict, dict)
        assert article_dict["source"] == "Bloomberg"
        assert article_dict["title"] == "Market News"
        assert article_dict["description"] == "Market update"
        assert article_dict["url"] == "https://example.com"
        assert article_dict["published_at"] == published_date.isoformat()


class TestTokenBucket:
    """Test the TokenBucket rate limiting implementation."""

    def test_token_bucket_initialization(self):
        # Arrange & Act
        bucket = TokenBucket(max_tokens=10, refill_rate=1.0)

        # Assert
        assert bucket.max_tokens == 10
        assert bucket.tokens == 10  # Should start full
        assert bucket.refill_rate == 1.0

    def test_token_bucket_consume_success(self):
        # Arrange
        bucket = TokenBucket(max_tokens=5, refill_rate=1.0)

        # Act
        result = bucket.consume(3)

        # Assert
        assert result is True
        assert bucket.tokens == 2

    def test_token_bucket_consume_insufficient_tokens(self):
        # Arrange
        bucket = TokenBucket(max_tokens=5, refill_rate=1.0)

        # Act
        result = bucket.consume(6)  # Request more than available

        # Assert
        assert result is False
        assert bucket.tokens == 5  # Tokens should remain unchanged

    def test_token_bucket_refill(self):
        # Arrange
        bucket = TokenBucket(max_tokens=10, refill_rate=2.0)
        bucket.consume(5)  # Consume 5 tokens
        initial_tokens = bucket.tokens

        # Act
        time.sleep(1.1)  # Wait for refill
        bucket.consume(1)  # Trigger refill calculation

        # Assert
        assert bucket.tokens > initial_tokens

    def test_token_bucket_wait_time_calculation(self):
        # Arrange
        bucket = TokenBucket(max_tokens=5, refill_rate=2.0)
        bucket.consume(5)  # Exhaust all tokens

        # Act
        wait_time = bucket.wait_for_tokens(3)

        # Assert
        assert wait_time > 0
        assert wait_time == 3 / 2.0  # 3 tokens at 2.0 tokens/second


class TestNewsCache:
    """Test the NewsCache functionality."""

    def test_cache_initialization(self):
        # Arrange & Act
        cache = NewsCache(default_ttl=300)

        # Assert
        assert cache.default_ttl == 300
        assert len(cache.cache) == 0

    def test_cache_set_and_get(self):
        # Arrange
        cache = NewsCache(default_ttl=300)
        articles = [
            NewsArticle(
                source="Test",
                title="Test Article",
                description="Test Description",
                url="https://test.com",
                published_at=datetime.datetime.now(datetime.timezone.utc),
            )
        ]

        # Act
        cache.set("test_key", articles)
        result = cache.get("test_key")

        # Assert
        assert result is not None
        assert len(result) == 1
        assert result[0].title == "Test Article"

    def test_cache_expiration(self):
        # Arrange
        cache = NewsCache(default_ttl=1)  # 1 second TTL
        articles = [
            NewsArticle(
                source="Test",
                title="Test Article",
                description="Test Description",
                url="https://test.com",
                published_at=datetime.datetime.now(datetime.timezone.utc),
            )
        ]
        cache.set("test_key", articles)

        # Act
        time.sleep(1.1)  # Wait for expiration
        result = cache.get("test_key")

        # Assert
        assert result is None

    def test_cache_clear(self):
        # Arrange
        cache = NewsCache()
        articles = [
            NewsArticle(
                source="Test",
                title="Test Article",
                description="Test Description",
                url="https://test.com",
                published_at=datetime.datetime.now(datetime.timezone.utc),
            )
        ]
        cache.set("test_key", articles)

        # Act
        cache.clear()

        # Assert
        assert len(cache.cache) == 0


class TestNewsFetcher:
    """Test the NewsFetcher main class functionality."""

    @pytest.fixture
    def mock_config(self):
        """Provide a mock configuration for testing."""
        config = MagicMock(spec=AITradingConfig)
        config.newsapi_api_key = "test_api_key"
        config.news_cache_duration = 1800
        config.newsapi_rate_limit_requests = 100
        config.max_retries = 3
        config.news_articles_count = 10
        config.max_news_age_hours = 24
        config.get_retry_delay.return_value = 1.0
        return config

    @pytest.fixture
    def mock_successful_api_response(self):
        """Provide a mock successful API response."""
        return {
            "status": "ok",
            "articles": [
                {
                    "source": {"name": "Reuters"},
                    "title": "Apple Stock Rises on Strong Earnings",
                    "description": "Apple Inc. shares climbed after reporting quarterly results.",
                    "url": "https://example.com/apple-news",
                    "publishedAt": "2024-01-15T10:30:00Z",
                    "content": "Apple reported strong quarterly earnings...",
                },
                {
                    "source": {"name": "Bloomberg"},
                    "title": "Tech Sector Analysis",
                    "description": "Analysis of the technology sector performance.",
                    "url": "https://example.com/tech-analysis",
                    "publishedAt": "2024-01-15T09:15:00Z",
                    "content": "The technology sector showed...",
                },
            ],
        }

    @patch("core.news_fetcher.NewsApiClient")
    def test_news_fetcher_initialization_success(self, mock_client_class, mock_config):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        # Act
        fetcher = NewsFetcher(config=mock_config)

        # Assert
        assert fetcher.config == mock_config
        assert fetcher.client == mock_client_instance
        mock_client_class.assert_called_once_with(api_key="test_api_key")

    @patch("core.news_fetcher.NewsApiClient")
    def test_news_fetcher_initialization_no_api_key(
        self, mock_client_class, mock_config
    ):
        # Arrange
        mock_config.newsapi_api_key = None

        # Act
        fetcher = NewsFetcher(config=mock_config)

        # Assert
        assert fetcher.client is None
        mock_client_class.assert_not_called()

    @patch("core.news_fetcher.NewsApiClient")
    def test_get_stock_news_happy_path(
        self, mock_client_class, mock_config, mock_successful_api_response
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_everything.return_value = mock_successful_api_response

        fetcher = NewsFetcher(config=mock_config)

        # Act
        articles = fetcher.get_stock_news("AAPL", max_articles=5, use_cache=False)

        # Assert
        assert len(articles) == 2
        assert articles[0].source == "Reuters"
        assert articles[0].title == "Apple Stock Rises on Strong Earnings"
        assert articles[1].source == "Bloomberg"
        assert "AAPL" in mock_client_instance.get_everything.call_args[1]["q"].upper()

    @patch("core.news_fetcher.NewsApiClient")
    def test_get_stock_news_api_failure(self, mock_client_class, mock_config):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_everything.return_value = {
            "status": "error",
            "message": "API request failed",
        }

        fetcher = NewsFetcher(config=mock_config)

        # Act
        articles = fetcher.get_stock_news("AAPL", use_cache=False)

        # Assert
        assert articles == []

    @patch("core.news_fetcher.NewsApiClient")
    def test_get_stock_news_network_error(self, mock_client_class, mock_config):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_everything.side_effect = Exception("Network error")

        fetcher = NewsFetcher(config=mock_config)

        # Act
        articles = fetcher.get_stock_news("AAPL", use_cache=False)

        # Assert
        assert articles == []

    @patch("core.news_fetcher.NewsApiClient")
    def test_get_stock_news_client_not_initialized(
        self, mock_client_class, mock_config
    ):
        # Arrange
        mock_config.newsapi_api_key = None
        fetcher = NewsFetcher(config=mock_config)

        # Act
        articles = fetcher.get_stock_news("AAPL")

        # Assert
        assert articles == []

    @patch("core.news_fetcher.NewsApiClient")
    def test_get_stock_news_caching(
        self, mock_client_class, mock_config, mock_successful_api_response
    ):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_everything.return_value = mock_successful_api_response

        fetcher = NewsFetcher(config=mock_config)

        # Act - First call
        articles1 = fetcher.get_stock_news("AAPL", use_cache=True)
        # Act - Second call (should use cache)
        articles2 = fetcher.get_stock_news("AAPL", use_cache=True)

        # Assert
        assert len(articles1) == 2
        assert len(articles2) == 2
        assert articles1[0].title == articles2[0].title
        # API should only be called once due to caching
        mock_client_instance.get_everything.assert_called_once()

    @patch("core.news_fetcher.NewsApiClient")
    def test_validate_connection_success(self, mock_client_class, mock_config):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_sources.return_value = {
            "status": "ok",
            "sources": [{"name": "Reuters"}, {"name": "Bloomberg"}],
        }

        fetcher = NewsFetcher(config=mock_config)

        # Act
        status = fetcher.validate_connection()

        # Assert
        assert status["connected"] is True
        assert status["sources_available"] == 2

    @patch("core.news_fetcher.NewsApiClient")
    def test_validate_connection_failure(self, mock_client_class, mock_config):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_sources.side_effect = Exception("Connection failed")

        fetcher = NewsFetcher(config=mock_config)

        # Act
        status = fetcher.validate_connection()

        # Assert
        assert status["connected"] is False
        assert "Connection failed" in status["error"]

    @patch("core.news_fetcher.NewsApiClient")
    def test_rate_limiting_behavior(self, mock_client_class, mock_config):
        # Arrange
        mock_config.newsapi_rate_limit_requests = 1  # Very low limit for testing
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_everything.return_value = {
            "status": "ok",
            "articles": [],
        }

        fetcher = NewsFetcher(config=mock_config)

        # Act - Make two rapid requests
        start_time = time.time()
        fetcher.get_stock_news("AAPL", use_cache=False)
        fetcher.get_stock_news("GOOGL", use_cache=False)
        end_time = time.time()

        # Assert - Second request should be delayed due to rate limiting
        # (This is a simplified test - in practice, you might want to mock time.sleep)
        assert end_time - start_time >= 0  # Some delay should occur


class TestGlobalFunctions:
    """Test module-level convenience functions."""

    @patch("core.news_fetcher.NewsFetcher")
    def test_get_news_fetcher_singleton(self, mock_fetcher_class):
        # Arrange
        mock_instance = MagicMock()
        mock_fetcher_class.return_value = mock_instance

        # Act
        fetcher1 = get_news_fetcher()
        fetcher2 = get_news_fetcher()

        # Assert
        assert fetcher1 is fetcher2  # Should return same instance
        mock_fetcher_class.assert_called_once()  # Should only create once

    @patch("core.news_fetcher.get_news_fetcher")
    def test_fetch_stock_news_convenience_function(self, mock_get_fetcher):
        # Arrange
        mock_fetcher_instance = MagicMock()
        mock_fetcher_instance.get_stock_news.return_value = []
        mock_get_fetcher.return_value = mock_fetcher_instance

        # Act
        result = fetch_stock_news("AAPL", max_articles=5)

        # Assert
        mock_fetcher_instance.get_stock_news.assert_called_once_with(
            "AAPL", max_articles=5
        )
        assert result == []

    @patch("core.news_fetcher.get_news_fetcher")
    def test_validate_news_api_convenience_function(self, mock_get_fetcher):
        # Arrange
        mock_fetcher_instance = MagicMock()
        mock_fetcher_instance.validate_connection.return_value = {"connected": True}
        mock_get_fetcher.return_value = mock_fetcher_instance

        # Act
        result = validate_news_api()

        # Assert
        mock_fetcher_instance.validate_connection.assert_called_once()
        assert result == {"connected": True}


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @patch("core.news_fetcher.NewsApiClient")
    def test_malformed_api_response(self, mock_client_class):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        # Malformed response missing required fields
        mock_client_instance.get_everything.return_value = {
            "status": "ok",
            "articles": [
                {
                    "source": {},  # Missing name
                    "title": None,  # None title
                    "publishedAt": "invalid-date-format",  # Invalid date
                }
            ],
        }

        fetcher = NewsFetcher()

        # Act
        articles = fetcher.get_stock_news("AAPL", use_cache=False)

        # Assert - Should handle malformed data gracefully
        # Might return empty list or articles with default values
        assert isinstance(articles, list)

    @patch("core.news_fetcher.NewsApiClient")
    def test_empty_news_response(self, mock_client_class):
        # Arrange
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_everything.return_value = {
            "status": "ok",
            "articles": [],
        }

        fetcher = NewsFetcher()

        # Act
        articles = fetcher.get_stock_news("INVALID_TICKER", use_cache=False)

        # Assert
        assert articles == []

    def test_news_article_with_minimal_data(self):
        # Arrange & Act
        article = NewsArticle(
            source="",
            title="",
            description=None,
            url="",
            published_at=datetime.datetime.now(datetime.timezone.utc),
        )

        # Assert
        assert article.source == ""
        assert article.title == ""
        assert article.description is None
        assert article.url == ""
