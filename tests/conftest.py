"""
Pytest Configuration and Central Fixtures

This file provides reusable test fixtures for the AI Trading Demo test suite.
Fixtures are used to reduce code duplication and provide consistent test data
across all test modules following the DRY principle.

Author: AI Trading Demo Team
Version: 1.0 (Hybrid Architecture)
"""

import pytest
import pandas as pd
import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

# Import our shared modules for testing
from shared.config import TradingConfig, get_config, reset_config
from shared.data_manager import DataManager
from shared.strategy import TradingStrategy
from shared.indicators import TechnicalIndicators


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

    # Add indicators using our TechnicalIndicators class
    df = TechnicalIndicators.add_all_indicators(df)

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
def trading_config() -> TradingConfig:
    """Provide a clean TradingConfig instance for testing.

    Returns:
        TradingConfig: A fresh configuration instance with default values.
    """
    return TradingConfig()


@pytest.fixture
def data_manager() -> DataManager:
    """Provide a DataManager instance for testing.

    Returns:
        DataManager: A DataManager instance with short cache duration for testing.
    """
    return DataManager(cache_duration=1)  # Very short cache for testing


@pytest.fixture
def trading_strategy() -> TradingStrategy:
    """Provide a TradingStrategy instance for testing.

    Returns:
        TradingStrategy: A strategy instance with default parameters.
    """
    return TradingStrategy(short_window=20, long_window=50)


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
