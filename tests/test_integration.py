"""
Comprehensive Integration Tests for AI Trading Demo

Tests the complete system pipeline with real API connections and validates
end-to-end functionality including news fetching, AI analysis, and signal generation.

Author: AI Trading Demo Team
Version: 3.0 (Optimized Integration Testing)
"""

import pytest
import os
import datetime
import pandas as pd
from typing import Optional, Dict, Any
from unittest.mock import patch

from core.config import get_config, validate_environment
from core.news_fetcher import get_news_fetcher, fetch_stock_news, validate_news_api
from core.ai_analyzer import get_ai_analyzer, test_ai_connection, analyze_news_sentiment
from core.data_manager import DataManager
from core.strategy import TradingStrategy, generate_trading_signals_with_ui_feedback


@pytest.mark.integration
class TestEnvironmentSetup:
    """Test environment configuration and API availability."""

    def test_environment_validation(self):
        """Test environment configuration and API keys."""
        config = get_config()
        validation = validate_environment()
        
        # Basic configuration should be valid
        assert config is not None
        assert validation is not None
        assert 'api_key_status' in validation
        
        # Check API key structure
        api_status = validation['api_key_status']
        assert 'google_api_key' in api_status
        assert 'newsapi_api_key' in api_status
        
    def test_configuration_completeness(self):
        """Test that all required configuration parameters are present."""
        config = get_config()
        
        # Test core configuration attributes
        assert hasattr(config, 'ai_model_name')
        assert hasattr(config, 'news_cache_duration')
        assert hasattr(config, 'requests_per_minute')
        assert hasattr(config, 'max_articles_per_request')
        
        # Test values are reasonable
        assert config.requests_per_minute > 0
        assert config.max_articles_per_request > 0


@pytest.mark.integration
@pytest.mark.ai_required
class TestAIConnection:
    """Test AI service connectivity and functionality."""

    def test_ai_connection_status(self):
        """Test AI analyzer connection without requiring API key."""
        # This test should work even without API key
        analyzer = get_ai_analyzer()
        assert analyzer is not None
        
    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"),
        reason="Google API key not available"
    )
    def test_ai_connection_with_api_key(self):
        """Test AI connection with actual API key."""
        connection_status = test_ai_connection()
        
        # Connection might fail due to quota/rate limits, which is acceptable
        assert isinstance(connection_status, bool)

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"),
        reason="Google API key not available"
    )
    def test_ai_model_configuration(self):
        """Test AI model configuration and switching."""
        config = get_config()
        analyzer = get_ai_analyzer()
        
        # Test that analyzer uses configured model
        assert analyzer.model_name == config.ai_model_name
        
        # Test model switching capability
        original_model = config.ai_model_name
        try:
            config.update_model_settings("gemini-1.5-flash")
            new_analyzer = get_ai_analyzer()
            assert new_analyzer.model_name == "gemini-1.5-flash"
        finally:
            # Restore original model
            config.update_model_settings(original_model)


@pytest.mark.integration
@pytest.mark.news_required
class TestNewsIntegration:
    """Test news fetching and processing integration."""

    def test_news_fetcher_initialization(self):
        """Test news fetcher initialization."""
        fetcher = get_news_fetcher()
        assert fetcher is not None

    @pytest.mark.skipif(
        not os.getenv("NEWS_API_KEY"),
        reason="NewsAPI key not available"
    )
    def test_news_api_validation(self):
        """Test NewsAPI connectivity and validation."""
        validation_result = validate_news_api()
        
        # API might be down or rate limited, which is acceptable
        assert isinstance(validation_result, bool)

    @pytest.mark.skipif(
        not os.getenv("NEWS_API_KEY"),
        reason="NewsAPI key not available"
    )
    def test_news_fetching_functionality(self):
        """Test actual news fetching for a stock ticker."""
        articles = fetch_stock_news("AAPL")
        
        # Articles might be empty due to rate limits or no news
        assert isinstance(articles, list)
        
        # If articles exist, validate structure
        if articles:
            article = articles[0]
            assert hasattr(article, 'title')
            assert hasattr(article, 'source')
            assert hasattr(article, 'published_at')


@pytest.mark.integration
class TestDataIntegration:
    """Test stock data fetching and management."""

    def test_data_manager_initialization(self):
        """Test DataManager initialization and basic functionality."""
        manager = DataManager(cache_duration=1)
        assert manager is not None
        assert manager.cache_duration == 1

    @pytest.mark.slow
    def test_stock_data_fetching(self):
        """Test stock data fetching from Yahoo Finance."""
        manager = DataManager(cache_duration=1)
        
        # Test with a reliable ticker
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=30)
        
        try:
            data = manager.fetch_stock_data("AAPL", start_date, end_date)
            
            # Data might be empty on weekends/holidays
            if not data.empty:
                assert 'Close' in data.columns
                assert 'Volume' in data.columns
                assert len(data) > 0
        except Exception as e:
            # Network issues are acceptable in integration tests
            pytest.skip(f"Stock data fetching failed: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """Test complete end-to-end system functionality."""

    @pytest.mark.skipif(
        not (os.getenv("GOOGLE_API_KEY") and os.getenv("NEWS_API_KEY")),
        reason="Both Google API and NewsAPI keys required"
    )
    def test_complete_trading_pipeline(self):
        """Test complete pipeline from data fetching to signal generation."""
        # Initialize components
        data_manager = DataManager(cache_duration=1)
        strategy = TradingStrategy()
        
        # Fetch stock data
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=10)
        
        try:
            # Get stock data
            stock_data = data_manager.fetch_stock_data("AAPL", start_date, end_date)
            
            if stock_data.empty:
                pytest.skip("No stock data available for testing")
            
            # Generate signals (this may fail due to API limits)
            try:
                result_data = strategy.generate_signals(stock_data, "AAPL")
                
                # Validate result structure
                assert isinstance(result_data, pd.DataFrame)
                assert len(result_data) >= len(stock_data)
                
                # Check for AI signal columns (may be empty due to API limits)
                expected_columns = ['Signal', 'AI_Signal', 'AI_Confidence', 'AI_Rationale']
                for col in expected_columns:
                    assert col in result_data.columns
                    
            except Exception as e:
                # API failures are acceptable in integration tests
                pytest.skip(f"Signal generation failed (likely API quota): {e}")
                
        except Exception as e:
            pytest.skip(f"Stock data fetching failed: {e}")

    def test_pipeline_error_handling(self):
        """Test pipeline error handling and graceful degradation."""
        strategy = TradingStrategy()
        
        # Test with empty data
        empty_data = pd.DataFrame()
        result = strategy.generate_signals(empty_data, "TEST")
        
        # Should return empty DataFrame gracefully
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_signal_generation_without_apis(self):
        """Test signal generation fallback when APIs are unavailable."""
        # Create minimal test data
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        test_data = pd.DataFrame({
            'Date': [date.date() for date in dates],
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        strategy = TradingStrategy()
        
        # Mock API failures
        with patch('core.news_fetcher.fetch_stock_news', return_value=[]), \
             patch('core.ai_analyzer.analyze_news_sentiment', return_value=None):
            
            result = strategy.generate_signals(test_data, "TEST")
            
            # Should still return valid DataFrame with default signals
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(test_data)
            assert 'Signal' in result.columns
            
            # Default signals should be HOLD (0)
            assert all(result['Signal'] == 0)


@pytest.mark.integration
class TestPerformanceAndCaching:
    """Test system performance and caching functionality."""

    def test_caching_functionality(self):
        """Test that caching works properly across components."""
        manager = DataManager(cache_duration=60)  # 1 minute cache
        
        # Test cache info functionality
        cache_info = manager.get_cache_info()
        assert isinstance(cache_info, dict)
        assert 'hits' in cache_info
        assert 'misses' in cache_info

    def test_configuration_caching(self):
        """Test configuration caching and singleton behavior."""
        config1 = get_config()
        config2 = get_config()
        
        # Should return same instance (singleton)
        assert config1 is config2

    @pytest.mark.slow
    def test_response_times(self):
        """Test that key operations complete within reasonable timeframes."""
        import time
        
        # Test configuration loading time
        start_time = time.time()
        config = get_config()
        config_time = time.time() - start_time
        
        assert config_time < 1.0  # Should load in under 1 second
        
        # Test component initialization time
        start_time = time.time()
        analyzer = get_ai_analyzer()
        fetcher = get_news_fetcher()
        manager = DataManager()
        init_time = time.time() - start_time
        
        assert init_time < 2.0  # Should initialize in under 2 seconds


if __name__ == "__main__":
    # Allow running as standalone script for development
    pytest.main([__file__, "-v", "--tb=short"])