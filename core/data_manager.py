"""
Data Manager Module for AI Trading Demo

This module provides centralized data management functionality shared between
Streamlit and NiceGUI applications. It handles data fetching, caching,
validation, and real-time data streaming coordination.

Author: AI Trading Demo Team
Version: 1.0 (Hybrid Architecture)
"""

import datetime
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from utils.errors import (
    setup_logger, handle_data_fetch_error
)
from ui.components import (
    display_streamlit_message, MessageType
)
from utils.api import APIValidator


class DataManager:
    """Centralized data management with caching and real-time capabilities."""

    def __init__(self, cache_duration: int = 300):
        """Initialize the DataManager with caching configuration.

        Args:
            cache_duration (int): Cache duration in seconds. Defaults to 300 (5 minutes).
        """
        self.cache_duration = cache_duration
        self._cache: Dict[str, Dict] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid based on timestamp.

        Args:
            cache_key (str): The cache key to check.

        Returns:
            bool: True if cache is valid, False otherwise.
        """
        if cache_key not in self._cache:
            return False

        cache_time = self._cache[cache_key]["timestamp"]
        return (time.time() - cache_time) < self.cache_duration

    def _get_cache_key(
        self, ticker: str, start_date: datetime.date, end_date: datetime.date
    ) -> str:
        """Generate a unique cache key for data requests.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (datetime.date): Start date for data.
            end_date (datetime.date): End date for data.

        Returns:
            str: Unique cache key.
        """
        return f"{ticker}_{start_date}_{end_date}"

    def fetch_stock_data(
        self, ticker: str, start_date: datetime.date, end_date: datetime.date
    ) -> pd.DataFrame:
        """Fetch historical stock data with caching support.

        This function handles the API request to Yahoo Finance, implements
        caching for performance, and provides comprehensive error handling.

        Args:
            ticker (str): Stock ticker symbol (e.g., "AAPL").
            start_date (datetime.date): Start date for historical data.
            end_date (datetime.date): End date for historical data.

        Returns:
            pd.DataFrame: OHLCV stock data with Date as a column.
                         Returns empty DataFrame if fetch fails.
        """
        try:
            # Generate cache key
            cache_key = self._get_cache_key(ticker, start_date, end_date)

            # Check cache first
            if self._is_cache_valid(cache_key):
                return self._cache[cache_key]["data"].copy()

            # Validate inputs
            if not ticker or not ticker.strip():
                raise ValueError("Ticker symbol cannot be empty")

            if start_date >= end_date:
                raise ValueError("Start date must be before end date")

            # Fetch data from Yahoo Finance
            stock = yf.Ticker(ticker.upper().strip())
            data = stock.history(start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")

            # Reset index to make Date a column
            data = data.reset_index()

            # Ensure Date column is datetime
            if "Date" in data.columns:
                data["Date"] = pd.to_datetime(data["Date"]).dt.date

            # Cache the result
            self._cache[cache_key] = {"data": data.copy(), "timestamp": time.time()}

            return data

        except Exception as e:
            # Use centralized error handling
            logger = setup_logger(__name__)
            error_info = handle_data_fetch_error(e, f"fetch stock data for {ticker}", logger)
            return pd.DataFrame()

    async def fetch_stock_data_async(
        self, ticker: str, start_date: datetime.date, end_date: datetime.date
    ) -> pd.DataFrame:
        """Asynchronously fetch stock data for real-time applications.

        This method is useful for NiceGUI applications that need non-blocking
        data fetching to maintain UI responsiveness.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (datetime.date): Start date for historical data.
            end_date (datetime.date): End date for historical data.

        Returns:
            pd.DataFrame: OHLCV stock data.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self.fetch_stock_data, ticker, start_date, end_date
        )

    def get_latest_price(self, ticker: str) -> Optional[Dict]:
        """Get the latest price and basic info for a ticker.

        This function fetches minimal real-time data useful for monitoring
        applications without the overhead of historical data.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            Optional[Dict]: Dictionary containing latest price info:
                           {'price', 'change', 'change_percent', 'volume', 'timestamp'}
                           Returns None if fetch fails.
        """
        try:
            if not ticker or not ticker.strip():
                return None

            stock = yf.Ticker(ticker.upper().strip())
            info = stock.fast_info

            if not info:
                return None

            # Get current price and change
            current_price = getattr(info, "last_price", None)
            previous_close = getattr(info, "previous_close", None)

            if current_price is None or previous_close is None:
                return None

            change = current_price - previous_close
            change_percent = (
                (change / previous_close * 100) if previous_close != 0 else 0
            )

            return {
                "price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "volume": getattr(info, "last_volume", 0),
                "timestamp": datetime.datetime.now(),
            }

        except Exception as e:
            # Use centralized error handling  
            logger = setup_logger(__name__)
            handle_data_fetch_error(e, f"fetch latest price for {ticker}", logger)
            return None

    async def get_latest_price_async(self, ticker: str) -> Optional[Dict]:
        """Asynchronously get latest price for real-time monitoring.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            Optional[Dict]: Latest price information.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.get_latest_price, ticker)

    def validate_ticker(self, ticker: str) -> bool:
        """Validate if a ticker symbol exists and has data available.

        This function performs a quick validation check useful for
        input validation in both applications.

        Args:
            ticker (str): Stock ticker symbol to validate.

        Returns:
            bool: True if ticker is valid, False otherwise.
        """
        try:
            if not ticker or not ticker.strip():
                return False

            stock = yf.Ticker(ticker.upper().strip())
            info = stock.fast_info

            # Check if we can get basic info
            return hasattr(info, "last_price") and info.last_price is not None

        except Exception:
            return False

    def validate_date_inputs(
        self, start_date: datetime.date, end_date: datetime.date
    ) -> Tuple[bool, Optional[str]]:
        """Validate date inputs for data fetching.

        This function provides comprehensive date validation with specific
        error messages for user feedback.

        Args:
            start_date (datetime.date): Start date to validate.
            end_date (datetime.date): End date to validate.

        Returns:
            tuple[bool, Optional[str]]: (is_valid, error_message)
                                      Returns (True, None) if valid.
        """
        try:
            today = datetime.date.today()

            # Check if start date is after end date
            if start_date >= end_date:
                return False, "Start date must be before end date"

            # Check if end date is in the future
            if end_date > today:
                return False, "End date cannot be in the future"

            # Check if date range is too old (market data availability)
            oldest_allowed = today - datetime.timedelta(days=365 * 20)  # 20 years
            if start_date < oldest_allowed:
                return False, "Start date is too far in the past (max 20 years)"

            # Check if date range is sufficient for analysis
            date_diff = (end_date - start_date).days
            if date_diff < 60:  # Need at least 60 days for 50-day SMA
                return False, "Date range too short (minimum 60 days required)"

            return True, None

        except Exception as e:
            return False, f"Date validation error: {e}"

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def get_cache_info(self) -> Dict:
        """Get information about the current cache state.

        Returns:
            Dict: Cache statistics and information.
        """
        return {
            "cache_size": len(self._cache),
            "cache_duration": self.cache_duration,
            "cache_keys": list(self._cache.keys()),
        }

    def cleanup(self) -> None:
        """Clean up resources including thread pool."""
        try:
            self._executor.shutdown(wait=True)
        except Exception:
            pass


# Additional utility functions moved from utils.py
def validate_date_inputs_ui(start_date: datetime.date, end_date: datetime.date) -> bool:
    """Validates user date inputs for logical consistency with UI feedback.

    This function ensures that the end date is not before the start date
    and provides appropriate user feedback through the Streamlit interface.
    This function is kept for UI-specific validation; use DataManager.validate_date_inputs
    for backend validation.

    Args:
        start_date (datetime.date): The start date for data fetching.
        end_date (datetime.date): The end date for data fetching.

    Returns:
        bool: True if dates are valid, False otherwise.
    """
    try:
        # Use centralized validation
        is_valid, error_msg = APIValidator.validate_date_range(start_date, end_date)
        
        if not is_valid:
            # Display validation error using streamlit message
            display_streamlit_message(
                MessageType.WARNING,
                "Date Validation Error",
                error_msg,
                "⚠️"
            )
            return False

        return True

    except Exception as e:
        display_streamlit_message(
            MessageType.ERROR,
            "Validation Error",
            f"Error validating date inputs: {str(e)}",
            "❌"
        )
        return False


def load_data_with_streamlit_cache(
    ticker: str, start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    """Fetches historical stock data with Streamlit caching.

    This function is a Streamlit-specific wrapper around DataManager functionality
    that provides UI feedback and caching specifically for Streamlit applications.

    Args:
        ticker (str): The stock ticker symbol to fetch (e.g., "AAPL").
        start_date (datetime.date): The start date for the historical data.
        end_date (datetime.date): The end date for the historical data.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing OHLCV data. Returns an
                      empty DataFrame if the download fails or no data is found.
    """
    try:
        import streamlit as st

        # Use DataManager for actual data fetching
        data_manager = DataManager()
        data = data_manager.fetch_stock_data(ticker, start_date, end_date)

        if data.empty:
            display_streamlit_message(
                MessageType.ERROR,
                "No Data Found",
                f"No data found for ticker '{ticker}' in the specified date range. Please verify the ticker symbol is correct and try a different date range.",
                "📊"
            )
            return pd.DataFrame()

        return data

    except Exception as e:
        try:
            import streamlit as st

            display_streamlit_message(
                MessageType.ERROR,
                f"Data Fetch Error for '{ticker}'",
                f"Error loading data: {str(e)}. Please verify the ticker symbol and check your network connectivity.",
                "❌"
            )
        except ImportError:
            print(f"Error loading data: {e}")
        return pd.DataFrame()
