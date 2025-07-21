"""
Tests for Data Manager Module

This module tests the robustness of data fetching, caching, validation, and error handling
for the DataManager class. All tests use pytest-mock to avoid real network requests following
the specifications in claude.md Chapter 11.5.3.

Author: AI Trading Demo Team
Version: 1.0 (Hybrid Architecture)
"""

import pytest
import pandas as pd
import datetime
import time
from unittest.mock import patch, MagicMock
from shared.data_manager import (
    DataManager,
    validate_date_inputs_ui,
    load_data_with_streamlit_cache,
)


class TestDataManager:
    """Test class for DataManager functionality."""

    def test_data_manager_initialization(self):
        """Test DataManager initialization with default and custom parameters."""
        # Arrange & Act: Create DataManager with default cache duration
        default_manager = DataManager()

        # Assert: Verify default cache duration
        assert (
            default_manager.cache_duration == 300
        ), "Default cache duration should be 300 seconds"

        # Arrange & Act: Create DataManager with custom cache duration
        custom_manager = DataManager(cache_duration=60)

        # Assert: Verify custom cache duration
        assert (
            custom_manager.cache_duration == 60
        ), "Custom cache duration should be 60 seconds"

    def test_fetch_stock_data_happy_path(self, mock_yfinance_success):
        """Test successful data fetching (Happy Path).

        Per Chapter 11.5.3: Mock a successful API call and verify correct handling.
        """
        # Arrange: Set up test parameters and DataManager
        data_manager = DataManager(cache_duration=1)
        ticker = "AAPL"
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Fetch stock data
        result_data = data_manager.fetch_stock_data(ticker, start_date, end_date)

        # Assert: Verify successful data fetching
        assert not result_data.empty, "Should return non-empty DataFrame"
        assert "Date" in result_data.columns, "Should have Date column"
        assert "Close" in result_data.columns, "Should have Close column"
        assert "Open" in result_data.columns, "Should have Open column"
        assert "High" in result_data.columns, "Should have High column"
        assert "Low" in result_data.columns, "Should have Low column"
        assert "Volume" in result_data.columns, "Should have Volume column"

        # Verify Date column is properly formatted
        assert all(
            isinstance(date, datetime.date) for date in result_data["Date"]
        ), "Date column should contain date objects"

        # Verify yfinance was called with correct parameters
        mock_yfinance_success.assert_called_once_with(ticker.upper())
        mock_instance = mock_yfinance_success.return_value
        mock_instance.history.assert_called_once_with(start=start_date, end=end_date)

    def test_fetch_stock_data_caching_functionality(self, mock_yfinance_success):
        """Test caching functionality works correctly."""
        # Arrange: Set up DataManager with short cache duration
        data_manager = DataManager(cache_duration=3600)  # 1 hour cache
        ticker = "AAPL"
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Fetch data twice
        first_result = data_manager.fetch_stock_data(ticker, start_date, end_date)
        second_result = data_manager.fetch_stock_data(ticker, start_date, end_date)

        # Assert: Should be identical and yfinance should only be called once (cached)
        pd.testing.assert_frame_equal(first_result, second_result)
        mock_yfinance_success.assert_called_once(), "Should only call API once due to caching"

    def test_fetch_stock_data_cache_expiry(self, mock_yfinance_success):
        """Test cache expiry functionality."""
        # Arrange: Set up DataManager with very short cache duration
        data_manager = DataManager(cache_duration=0.1)  # 0.1 seconds
        ticker = "AAPL"
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Fetch data, wait for cache to expire, fetch again
        first_result = data_manager.fetch_stock_data(ticker, start_date, end_date)
        time.sleep(0.2)  # Wait for cache to expire
        second_result = data_manager.fetch_stock_data(ticker, start_date, end_date)

        # Assert: Should call API twice due to cache expiry
        assert (
            mock_yfinance_success.call_count == 2
        ), "Should call API twice after cache expiry"
        pd.testing.assert_frame_equal(first_result, second_result)

    def test_fetch_stock_data_empty_response(self, mock_yfinance_empty):
        """Test handling of empty data from yfinance.

        Per Chapter 11.5.3: Mock yfinance to return empty DataFrame.
        """
        # Arrange: Set up DataManager and test parameters
        data_manager = DataManager()
        ticker = "INVALID"
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Try to fetch data
        result_data = data_manager.fetch_stock_data(ticker, start_date, end_date)

        # Assert: Should return empty DataFrame
        assert result_data.empty, "Should return empty DataFrame for empty API response"

        # Verify yfinance was called
        mock_yfinance_empty.assert_called_once_with(ticker.upper())

    def test_fetch_stock_data_api_exception(self, mock_yfinance_error):
        """Test handling of API exceptions.

        Per Chapter 11.5.3: Mock yfinance to raise an exception.
        """
        # Arrange: Set up DataManager and test parameters
        data_manager = DataManager()
        ticker = "AAPL"
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Try to fetch data
        result_data = data_manager.fetch_stock_data(ticker, start_date, end_date)

        # Assert: Should return empty DataFrame and not crash
        assert (
            result_data.empty
        ), "Should return empty DataFrame when API raises exception"

        # Verify yfinance was called
        mock_yfinance_error.assert_called_once_with(ticker.upper())

    def test_fetch_stock_data_input_validation_empty_ticker(self):
        """Test input validation with empty ticker."""
        # Arrange: DataManager and invalid inputs
        data_manager = DataManager()
        empty_ticker = ""
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Try to fetch data with empty ticker
        result_data = data_manager.fetch_stock_data(empty_ticker, start_date, end_date)

        # Assert: Should return empty DataFrame
        assert result_data.empty, "Should return empty DataFrame for empty ticker"

    def test_fetch_stock_data_input_validation_invalid_dates(self):
        """Test input validation with invalid date range."""
        # Arrange: DataManager and invalid date range
        data_manager = DataManager()
        ticker = "AAPL"
        start_date = datetime.date(2023, 1, 5)
        end_date = datetime.date(2023, 1, 1)  # End before start

        # Act: Try to fetch data with invalid dates
        result_data = data_manager.fetch_stock_data(ticker, start_date, end_date)

        # Assert: Should return empty DataFrame
        assert result_data.empty, "Should return empty DataFrame for invalid date range"

    def test_fetch_stock_data_ticker_normalization(self, mock_yfinance_success):
        """Test that ticker symbols are properly normalized (uppercase, stripped)."""
        # Arrange: DataManager and ticker with spaces and lowercase
        data_manager = DataManager()
        ticker = " aapl "  # Lowercase with spaces
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Fetch stock data
        result_data = data_manager.fetch_stock_data(ticker, start_date, end_date)

        # Assert: Should call yfinance with normalized ticker
        mock_yfinance_success.assert_called_once_with("AAPL")
        assert (
            not result_data.empty
        ), "Should successfully fetch data with normalized ticker"

    @pytest.mark.asyncio
    async def test_fetch_stock_data_async_happy_path(self, mock_yfinance_success):
        """Test asynchronous data fetching functionality."""
        # Arrange: DataManager and test parameters
        data_manager = DataManager()
        ticker = "AAPL"
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Fetch stock data asynchronously
        result_data = await data_manager.fetch_stock_data_async(
            ticker, start_date, end_date
        )

        # Assert: Should return same result as synchronous version
        assert not result_data.empty, "Should return non-empty DataFrame"
        assert "Date" in result_data.columns, "Should have Date column"
        mock_yfinance_success.assert_called_once_with(ticker.upper())

    def test_get_latest_price_happy_path(self):
        """Test getting latest price information successfully."""
        # Arrange: Mock yfinance fast_info
        data_manager = DataManager()
        ticker = "AAPL"

        with patch("yfinance.Ticker") as mock_ticker:
            # Set up mock fast_info
            mock_fast_info = MagicMock()
            mock_fast_info.last_price = 150.0
            mock_fast_info.previous_close = 145.0
            mock_fast_info.last_volume = 1000000

            mock_instance = MagicMock()
            mock_instance.fast_info = mock_fast_info
            mock_ticker.return_value = mock_instance

            # Act: Get latest price
            price_info = data_manager.get_latest_price(ticker)

            # Assert: Verify price information structure and calculations
            assert price_info is not None, "Should return price information"
            assert price_info["price"] == 150.0, "Should return correct current price"
            assert price_info["change"] == 5.0, "Should calculate correct price change"
            assert (
                abs(price_info["change_percent"] - 3.45) < 0.01
            ), "Should calculate correct change percentage"
            assert price_info["volume"] == 1000000, "Should return correct volume"
            assert "timestamp" in price_info, "Should include timestamp"

    def test_get_latest_price_invalid_ticker(self):
        """Test getting latest price with invalid ticker."""
        # Arrange: Mock yfinance to return None for fast_info
        data_manager = DataManager()
        ticker = "INVALID"

        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.fast_info = None
            mock_ticker.return_value = mock_instance

            # Act: Try to get latest price
            price_info = data_manager.get_latest_price(ticker)

            # Assert: Should return None
            assert price_info is None, "Should return None for invalid ticker"

    def test_get_latest_price_empty_ticker(self):
        """Test getting latest price with empty ticker."""
        # Arrange: DataManager and empty ticker
        data_manager = DataManager()
        empty_ticker = ""

        # Act: Try to get latest price
        price_info = data_manager.get_latest_price(empty_ticker)

        # Assert: Should return None
        assert price_info is None, "Should return None for empty ticker"

    @pytest.mark.asyncio
    async def test_get_latest_price_async(self):
        """Test asynchronous latest price fetching."""
        # Arrange: Mock yfinance
        data_manager = DataManager()
        ticker = "AAPL"

        with patch("yfinance.Ticker") as mock_ticker:
            mock_fast_info = MagicMock()
            mock_fast_info.last_price = 150.0
            mock_fast_info.previous_close = 145.0
            mock_fast_info.last_volume = 1000000

            mock_instance = MagicMock()
            mock_instance.fast_info = mock_fast_info
            mock_ticker.return_value = mock_instance

            # Act: Get latest price asynchronously
            price_info = await data_manager.get_latest_price_async(ticker)

            # Assert: Should return same result as synchronous version
            assert price_info is not None, "Should return price information"
            assert price_info["price"] == 150.0, "Should return correct current price"

    def test_validate_ticker_valid_ticker(self):
        """Test ticker validation with valid ticker."""
        # Arrange: Mock yfinance with valid ticker
        data_manager = DataManager()
        ticker = "AAPL"

        with patch("yfinance.Ticker") as mock_ticker:
            mock_fast_info = MagicMock()
            mock_fast_info.last_price = 150.0

            mock_instance = MagicMock()
            mock_instance.fast_info = mock_fast_info
            mock_ticker.return_value = mock_instance

            # Act: Validate ticker
            is_valid = data_manager.validate_ticker(ticker)

            # Assert: Should return True for valid ticker
            assert is_valid is True, "Should return True for valid ticker"

    def test_validate_ticker_invalid_ticker(self):
        """Test ticker validation with invalid ticker."""
        # Arrange: Mock yfinance with invalid ticker
        data_manager = DataManager()
        ticker = "INVALID"

        with patch("yfinance.Ticker") as mock_ticker:
            mock_fast_info = MagicMock()
            mock_fast_info.last_price = None

            mock_instance = MagicMock()
            mock_instance.fast_info = mock_fast_info
            mock_ticker.return_value = mock_instance

            # Act: Validate ticker
            is_valid = data_manager.validate_ticker(ticker)

            # Assert: Should return False for invalid ticker
            assert is_valid is False, "Should return False for invalid ticker"

    def test_validate_ticker_empty_ticker(self):
        """Test ticker validation with empty ticker."""
        # Arrange: DataManager and empty ticker
        data_manager = DataManager()
        empty_ticker = ""

        # Act: Validate empty ticker
        is_valid = data_manager.validate_ticker(empty_ticker)

        # Assert: Should return False
        assert is_valid is False, "Should return False for empty ticker"

    def test_validate_date_inputs_valid_dates(self, valid_date_range):
        """Test date validation with valid date range."""
        # Arrange: DataManager and valid dates
        data_manager = DataManager()
        start_date = valid_date_range["start_date"]
        end_date = valid_date_range["end_date"]

        # Act: Validate dates
        is_valid, error_message = data_manager.validate_date_inputs(
            start_date, end_date
        )

        # Assert: Should return True with no error message
        assert is_valid is True, "Should return True for valid dates"
        assert error_message is None, "Should not return error message for valid dates"

    def test_validate_date_inputs_invalid_range(self, invalid_date_range):
        """Test date validation with invalid date range."""
        # Arrange: DataManager and invalid dates (start after end)
        data_manager = DataManager()
        start_date = invalid_date_range["start_date"]
        end_date = invalid_date_range["end_date"]

        # Act: Validate dates
        is_valid, error_message = data_manager.validate_date_inputs(
            start_date, end_date
        )

        # Assert: Should return False with error message
        assert is_valid is False, "Should return False for invalid date range"
        assert (
            "Start date must be before end date" in error_message
        ), "Should return appropriate error message"

    def test_validate_date_inputs_future_end_date(self):
        """Test date validation with future end date."""
        # Arrange: DataManager and future end date
        data_manager = DataManager()
        start_date = datetime.date.today() - datetime.timedelta(days=30)
        end_date = datetime.date.today() + datetime.timedelta(days=10)  # Future date

        # Act: Validate dates
        is_valid, error_message = data_manager.validate_date_inputs(
            start_date, end_date
        )

        # Assert: Should return False with error message
        assert is_valid is False, "Should return False for future end date"
        assert (
            "End date cannot be in the future" in error_message
        ), "Should return appropriate error message"

    def test_validate_date_inputs_insufficient_range(self):
        """Test date validation with insufficient date range for analysis."""
        # Arrange: DataManager and very short date range
        data_manager = DataManager()
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=30)  # Only 30 days (need 60)

        # Act: Validate dates
        is_valid, error_message = data_manager.validate_date_inputs(
            start_date, end_date
        )

        # Assert: Should return False with error message
        assert is_valid is False, "Should return False for insufficient date range"
        assert (
            "Date range too short" in error_message
        ), "Should return appropriate error message"

    def test_clear_cache_functionality(self, mock_yfinance_success):
        """Test cache clearing functionality."""
        # Arrange: DataManager with cached data
        data_manager = DataManager(cache_duration=3600)
        ticker = "AAPL"
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Cache some data
        data_manager.fetch_stock_data(ticker, start_date, end_date)
        cache_info_before = data_manager.get_cache_info()

        # Act: Clear cache
        data_manager.clear_cache()
        cache_info_after = data_manager.get_cache_info()

        # Assert: Cache should be cleared
        assert (
            cache_info_before["cache_size"] > 0
        ), "Should have cached data before clearing"
        assert (
            cache_info_after["cache_size"] == 0
        ), "Should have empty cache after clearing"

    def test_get_cache_info(self, mock_yfinance_success):
        """Test cache information retrieval."""
        # Arrange: DataManager
        data_manager = DataManager()

        # Get initial cache info
        initial_info = data_manager.get_cache_info()

        # Cache some data
        data_manager.fetch_stock_data(
            "AAPL", datetime.date(2023, 1, 1), datetime.date(2023, 1, 3)
        )
        updated_info = data_manager.get_cache_info()

        # Assert: Cache info should reflect changes
        assert initial_info["cache_size"] == 0, "Initial cache should be empty"
        assert (
            updated_info["cache_size"] == 1
        ), "Cache should have one entry after fetching"
        assert "cache_duration" in updated_info, "Should include cache duration"
        assert "cache_keys" in updated_info, "Should include cache keys"

    def test_cleanup_functionality(self):
        """Test cleanup functionality."""
        # Arrange: DataManager with ThreadPoolExecutor
        data_manager = DataManager()

        # Act & Assert: Should not raise exception
        data_manager.cleanup()
        # If no exception is raised, cleanup worked properly


class TestDataManagerUIFunctions:
    """Test UI-specific data manager functions for Streamlit integration."""

    def test_validate_date_inputs_ui_valid_dates(
        self, valid_date_range, mock_streamlit
    ):
        """Test UI date validation with valid dates."""
        # Arrange: Valid date range
        start_date = valid_date_range["start_date"]
        end_date = valid_date_range["end_date"]

        # Act: Validate dates with UI feedback
        is_valid = validate_date_inputs_ui(start_date, end_date)

        # Assert: Should return True for valid dates
        assert is_valid is True, "Should return True for valid dates"

    def test_validate_date_inputs_ui_invalid_dates(
        self, invalid_date_range, mock_streamlit
    ):
        """Test UI date validation with invalid dates."""
        # Arrange: Invalid date range
        start_date = invalid_date_range["start_date"]
        end_date = invalid_date_range["end_date"]

        # Act: Validate dates with UI feedback
        is_valid = validate_date_inputs_ui(start_date, end_date)

        # Assert: Should return False for invalid dates
        assert is_valid is False, "Should return False for invalid dates"

    def test_validate_date_inputs_ui_future_start_date(self, mock_streamlit):
        """Test UI date validation with future start date."""
        # Arrange: Future start date
        start_date = datetime.date.today() + datetime.timedelta(days=10)
        end_date = datetime.date.today() + datetime.timedelta(days=20)

        # Act: Validate dates with UI feedback
        is_valid = validate_date_inputs_ui(start_date, end_date)

        # Assert: Should return False
        assert is_valid is False, "Should return False for future start date"

    def test_load_data_with_streamlit_cache_happy_path(
        self, mock_yfinance_success, mock_streamlit
    ):
        """Test Streamlit-cached data loading with successful response."""
        # Arrange: Test parameters
        ticker = "AAPL"
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Load data with Streamlit caching
        result_data = load_data_with_streamlit_cache(ticker, start_date, end_date)

        # Assert: Should return valid data
        assert not result_data.empty, "Should return non-empty DataFrame"
        assert "Date" in result_data.columns, "Should have Date column"
        assert "Close" in result_data.columns, "Should have Close column"

    def test_load_data_with_streamlit_cache_empty_response(
        self, mock_yfinance_empty, mock_streamlit
    ):
        """Test Streamlit-cached data loading with empty response."""
        # Arrange: Test parameters
        ticker = "INVALID"
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Try to load data
        result_data = load_data_with_streamlit_cache(ticker, start_date, end_date)

        # Assert: Should return empty DataFrame
        assert result_data.empty, "Should return empty DataFrame for empty response"

    def test_load_data_with_streamlit_cache_api_error(
        self, mock_yfinance_error, mock_streamlit
    ):
        """Test Streamlit-cached data loading with API error."""
        # Arrange: Test parameters
        ticker = "AAPL"
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Try to load data
        result_data = load_data_with_streamlit_cache(ticker, start_date, end_date)

        # Assert: Should return empty DataFrame and not crash
        assert (
            result_data.empty
        ), "Should return empty DataFrame when API raises exception"

    def test_load_data_with_streamlit_cache_limited_data_warning(self, mock_streamlit):
        """Test Streamlit warning for limited data."""
        # Arrange: Mock yfinance to return limited data
        limited_data = pd.DataFrame(
            {
                "Open": [100.0, 102.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 101.0],
                "Close": [102.0, 104.0],
                "Volume": [1000000, 1100000],
            }
        )
        limited_data.index = pd.date_range("2023-01-01", periods=2, freq="D")

        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            limited_data.index.name = "Date"
            mock_instance.history.return_value = limited_data
            mock_ticker.return_value = mock_instance

            # Act: Load limited data
            result_data = load_data_with_streamlit_cache(
                "AAPL", datetime.date(2023, 1, 1), datetime.date(2023, 1, 2)
            )

            # Assert: Should return data but with warning
            assert not result_data.empty, "Should return data even if limited"
            assert len(result_data) == 2, "Should have 2 rows of data"


class TestDataManagerEdgeCases:
    """Test edge cases and error scenarios for DataManager."""

    def test_cache_key_generation_consistency(self):
        """Test that cache keys are generated consistently."""
        # Arrange: DataManager
        data_manager = DataManager()
        ticker = "AAPL"
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Act: Generate cache key multiple times
        key1 = data_manager._get_cache_key(ticker, start_date, end_date)
        key2 = data_manager._get_cache_key(ticker, start_date, end_date)

        # Assert: Should be identical
        assert key1 == key2, "Cache keys should be consistent"
        assert ticker in key1, "Cache key should contain ticker"
        assert str(start_date) in key1, "Cache key should contain start date"
        assert str(end_date) in key1, "Cache key should contain end date"

    def test_cache_validation_edge_cases(self):
        """Test cache validation with edge cases."""
        # Arrange: DataManager
        data_manager = DataManager(cache_duration=1)

        # Test non-existent cache key
        is_valid = data_manager._is_cache_valid("non-existent-key")
        assert is_valid is False, "Non-existent cache key should be invalid"

        # Test expired cache
        data_manager._cache["test-key"] = {
            "data": pd.DataFrame(),
            "timestamp": time.time() - 3600,  # 1 hour ago
        }
        is_valid = data_manager._is_cache_valid("test-key")
        assert is_valid is False, "Expired cache should be invalid"

    def test_date_validation_edge_cases(self):
        """Test date validation with various edge cases."""
        # Arrange: DataManager
        data_manager = DataManager()

        # Test very old start date
        very_old_start = datetime.date(1900, 1, 1)
        today = datetime.date.today()
        is_valid, error = data_manager.validate_date_inputs(very_old_start, today)
        assert is_valid is False, "Very old start date should be invalid"
        assert "too far in the past" in error, "Should mention date is too old"

        # Test exact minimum date range (60 days)
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=60)
        is_valid, error = data_manager.validate_date_inputs(start_date, end_date)
        assert is_valid is True, "Exact minimum range should be valid"

        # Test one day short of minimum (59 days)
        start_date = end_date - datetime.timedelta(days=59)
        is_valid, error = data_manager.validate_date_inputs(start_date, end_date)
        assert is_valid is False, "One day short should be invalid"

    def test_price_change_calculation_edge_cases(self):
        """Test price change calculation with edge cases."""
        # Arrange: DataManager
        data_manager = DataManager()

        with patch("yfinance.Ticker") as mock_ticker:
            # Test zero previous close (division by zero)
            mock_fast_info = MagicMock()
            mock_fast_info.last_price = 100.0
            mock_fast_info.previous_close = 0.0
            mock_fast_info.last_volume = 1000000

            mock_instance = MagicMock()
            mock_instance.fast_info = mock_fast_info
            mock_ticker.return_value = mock_instance

            # Act: Get latest price with zero previous close
            price_info = data_manager.get_latest_price("TEST")

            # Assert: Should handle division by zero gracefully
            assert (
                price_info is not None
            ), "Should return result even with zero previous close"
            assert (
                price_info["change_percent"] == 0
            ), "Change percent should be 0 for zero previous close"
