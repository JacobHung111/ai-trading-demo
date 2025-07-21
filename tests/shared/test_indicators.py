"""
Tests for Technical Indicators Module

This module tests the mathematical correctness of all indicator calculations
including SMA calculations, edge case handling, and data validation following
the AAA (Arrange, Act, Assert) pattern.

Author: AI Trading Demo Team
Version: 1.0 (Hybrid Architecture)
"""

import pytest
import pandas as pd
import numpy as np
from shared.indicators import TechnicalIndicators, calculate_indicators_with_ui_feedback


class TestTechnicalIndicators:
    """Test class for TechnicalIndicators functionality."""

    def test_calculate_sma_with_valid_data(self, sample_ohlcv_data):
        """Test SMA calculation with valid data following AAA pattern.

        This test verifies that the SMA calculation produces mathematically
        correct results using a known dataset.
        """
        # Arrange: Set up test conditions with sample data
        close_prices = sample_ohlcv_data["Close"]
        window = 5

        # Act: Execute the SMA calculation
        result = TechnicalIndicators.calculate_sma(close_prices, window)

        # Assert: Verify the results are correct
        assert len(result) == len(
            close_prices
        ), "Result length should match input length"

        # Check that initial values are NaN
        for i in range(window - 1):
            assert pd.isna(result.iloc[i]), f"Position {i} should be NaN"

        # Check specific calculated values (manually verified)
        # For positions 4 onwards, verify SMA calculation
        expected_sma_5 = (102.0 + 104.0 + 106.0 + 108.0 + 110.0) / 5  # 106.0
        assert (
            abs(result.iloc[4] - expected_sma_5) < 0.001
        ), "SMA calculation incorrect at position 4"

        # Verify another calculation point
        expected_sma_10 = sample_ohlcv_data["Close"].iloc[5:10].mean()
        assert (
            abs(result.iloc[9] - expected_sma_10) < 0.001
        ), "SMA calculation incorrect at position 9"

    def test_calculate_sma_insufficient_data(self):
        """Test SMA calculation with insufficient data (edge case).

        This test ensures the function handles cases where the data
        has fewer rows than the window size gracefully.
        """
        # Arrange: Create data with fewer points than window
        insufficient_data = pd.Series([100.0, 102.0, 104.0])
        window = 5  # More than data length

        # Act: Calculate SMA with insufficient data
        result = TechnicalIndicators.calculate_sma(insufficient_data, window)

        # Assert: All values should be NaN
        assert len(result) == len(
            insufficient_data
        ), "Result length should match input length"
        assert all(pd.isna(result)), "All values should be NaN when data insufficient"

    def test_calculate_sma_empty_data(self):
        """Test SMA calculation with empty data."""
        # Arrange: Empty series
        empty_data = pd.Series([])
        window = 5

        # Act: Calculate SMA with empty data
        result = TechnicalIndicators.calculate_sma(empty_data, window)

        # Assert: Result should be empty
        assert len(result) == 0, "Empty input should produce empty output"

    def test_calculate_sma20_convenience_method(self, sample_ohlcv_data):
        """Test the SMA20 convenience method."""
        # Arrange: Use sample close prices
        close_prices = sample_ohlcv_data["Close"]

        # Act: Calculate using convenience method and direct method
        sma20_convenience = TechnicalIndicators.calculate_sma20(close_prices)
        sma20_direct = TechnicalIndicators.calculate_sma(close_prices, 20)

        # Assert: Both methods should produce identical results
        pd.testing.assert_series_equal(sma20_convenience, sma20_direct)

        # Verify that first 19 values are NaN
        assert sum(pd.isna(sma20_convenience)) >= 19, "First 19 values should be NaN"

    def test_calculate_sma50_convenience_method(self, sample_ohlcv_data):
        """Test the SMA50 convenience method."""
        # Arrange: Use sample close prices
        close_prices = sample_ohlcv_data["Close"]

        # Act: Calculate using convenience method and direct method
        sma50_convenience = TechnicalIndicators.calculate_sma50(close_prices)
        sma50_direct = TechnicalIndicators.calculate_sma(close_prices, 50)

        # Assert: Both methods should produce identical results
        pd.testing.assert_series_equal(sma50_convenience, sma50_direct)

        # Verify that first 49 values are NaN
        assert sum(pd.isna(sma50_convenience)) >= 49, "First 49 values should be NaN"

    def test_add_all_indicators_valid_data(self, sample_ohlcv_data):
        """Test adding all indicators to valid OHLCV data."""
        # Arrange: Use sample OHLCV data
        input_data = sample_ohlcv_data.copy()
        original_columns = set(input_data.columns)

        # Act: Add all indicators
        result_data = TechnicalIndicators.add_all_indicators(input_data)

        # Assert: Verify new columns are added
        assert "SMA20" in result_data.columns, "SMA20 column should be added"
        assert "SMA50" in result_data.columns, "SMA50 column should be added"
        assert len(result_data) == len(input_data), "Row count should be preserved"

        # Verify original columns are preserved
        for col in original_columns:
            assert (
                col in result_data.columns
            ), f"Original column {col} should be preserved"

        # Verify SMA calculations are correct
        expected_sma20 = TechnicalIndicators.calculate_sma20(input_data["Close"])
        expected_sma50 = TechnicalIndicators.calculate_sma50(input_data["Close"])

        pd.testing.assert_series_equal(
            result_data["SMA20"], expected_sma20, check_names=False
        )
        pd.testing.assert_series_equal(
            result_data["SMA50"], expected_sma50, check_names=False
        )

    def test_add_all_indicators_missing_close_column(self):
        """Test adding indicators with missing Close column (error case)."""
        # Arrange: Data without Close column
        invalid_data = pd.DataFrame(
            {
                "Open": [100.0, 102.0, 104.0],
                "High": [105.0, 106.0, 108.0],
                "Low": [99.0, 101.0, 103.0],
                "Volume": [1000, 1100, 1200],
            }
        )

        # Act: Try to add indicators
        result_data = TechnicalIndicators.add_all_indicators(invalid_data)

        # Assert: Should return empty DataFrame due to error
        assert (
            result_data.empty
        ), "Should return empty DataFrame when Close column missing"

    def test_add_all_indicators_custom_price_column(self, sample_ohlcv_data):
        """Test adding indicators using a custom price column."""
        # Arrange: Use 'Open' instead of 'Close'
        input_data = sample_ohlcv_data.copy()

        # Act: Add indicators using Open prices
        result_data = TechnicalIndicators.add_all_indicators(
            input_data, price_column="Open"
        )

        # Assert: Verify calculations are based on Open prices
        expected_sma20_open = TechnicalIndicators.calculate_sma20(input_data["Open"])
        expected_sma50_open = TechnicalIndicators.calculate_sma50(input_data["Open"])

        pd.testing.assert_series_equal(
            result_data["SMA20"], expected_sma20_open, check_names=False
        )
        pd.testing.assert_series_equal(
            result_data["SMA50"], expected_sma50_open, check_names=False
        )

    def test_get_latest_indicators_valid_data(self, sample_with_indicators):
        """Test getting latest indicator values from valid data."""
        # Arrange: Use data with indicators already calculated
        data_with_indicators = sample_with_indicators.copy()

        # Act: Get latest indicators
        latest_indicators = TechnicalIndicators.get_latest_indicators(
            data_with_indicators
        )

        # Assert: Verify result structure and values
        assert latest_indicators is not None, "Should return valid result"
        assert "price" in latest_indicators, "Should include current price"
        assert "sma20" in latest_indicators, "Should include SMA20"
        assert "sma50" in latest_indicators, "Should include SMA50"
        assert "date" in latest_indicators, "Should include date"

        # Verify values are numeric
        assert isinstance(latest_indicators["price"], float), "Price should be float"
        assert isinstance(latest_indicators["sma20"], float), "SMA20 should be float"
        assert isinstance(latest_indicators["sma50"], float), "SMA50 should be float"

    def test_get_latest_indicators_insufficient_data(self, sample_minimal_data):
        """Test getting latest indicators with insufficient data."""
        # Arrange: Use minimal data (not enough for SMA50)
        minimal_data = sample_minimal_data.copy()
        # Add indicators (will be mostly NaN)
        minimal_with_indicators = TechnicalIndicators.add_all_indicators(minimal_data)

        # Act: Try to get latest indicators
        latest_indicators = TechnicalIndicators.get_latest_indicators(
            minimal_with_indicators
        )

        # Assert: Should return None due to insufficient data
        assert latest_indicators is None, "Should return None with insufficient data"

    def test_get_latest_indicators_empty_data(self):
        """Test getting latest indicators from empty data."""
        # Arrange: Empty DataFrame
        empty_data = pd.DataFrame()

        # Act: Try to get latest indicators
        latest_indicators = TechnicalIndicators.get_latest_indicators(empty_data)

        # Assert: Should return None
        assert latest_indicators is None, "Should return None for empty data"

    def test_validate_indicator_data_valid(self, sample_with_indicators):
        """Test validation with valid indicator data."""
        # Arrange: Use data with valid indicators
        valid_data = sample_with_indicators.copy()

        # Act: Validate the data
        is_valid = TechnicalIndicators.validate_indicator_data(valid_data)

        # Assert: Should be valid
        assert is_valid is True, "Valid indicator data should pass validation"

    def test_validate_indicator_data_missing_columns(self, sample_ohlcv_data):
        """Test validation with missing indicator columns."""
        # Arrange: Data without indicator columns
        data_without_indicators = sample_ohlcv_data.copy()

        # Act: Validate the data
        is_valid = TechnicalIndicators.validate_indicator_data(data_without_indicators)

        # Assert: Should be invalid
        assert is_valid is False, "Data without indicators should fail validation"

    def test_validate_indicator_data_insufficient_rows(self):
        """Test validation with insufficient data rows."""
        # Arrange: Data with indicators but only one row
        insufficient_data = pd.DataFrame({"SMA20": [100.0], "SMA50": [101.0]})

        # Act: Validate the data
        is_valid = TechnicalIndicators.validate_indicator_data(insufficient_data)

        # Assert: Should be invalid (need at least 2 rows for crossover detection)
        assert (
            is_valid is False
        ), "Single row should be insufficient for signal generation"

    def test_validate_indicator_data_with_nan_values(self):
        """Test validation with NaN values in indicators."""
        # Arrange: Data with NaN values
        data_with_nans = pd.DataFrame(
            {
                "SMA20": [np.nan, np.nan, 100.0, 101.0],
                "SMA50": [np.nan, np.nan, 99.0, 100.0],
            }
        )

        # Act: Validate the data
        is_valid = TechnicalIndicators.validate_indicator_data(data_with_nans)

        # Assert: Should still be valid (has 2 non-NaN rows)
        assert (
            is_valid is True
        ), "Data with some NaN values but sufficient non-NaN rows should be valid"


class TestIndicatorsUIFeedbackFunctions:
    """Test UI feedback functions for Streamlit integration."""

    def test_calculate_indicators_with_ui_feedback_valid_data(
        self, sample_ohlcv_data, mock_streamlit
    ):
        """Test indicator calculation with UI feedback using valid data."""
        # Arrange: Use sample OHLCV data
        input_data = sample_ohlcv_data.copy()

        # Act: Calculate indicators with UI feedback
        result_data = calculate_indicators_with_ui_feedback(input_data)

        # Assert: Should return data with indicators
        assert "SMA20" in result_data.columns, "SMA20 should be added"
        assert "SMA50" in result_data.columns, "SMA50 should be added"
        assert len(result_data) == len(input_data), "Row count should be preserved"

    def test_calculate_indicators_with_ui_feedback_missing_close(self, mock_streamlit):
        """Test UI feedback function with missing Close column."""
        # Arrange: Data without Close column
        invalid_data = pd.DataFrame(
            {"Open": [100.0, 102.0, 104.0], "High": [105.0, 106.0, 108.0]}
        )

        # Act: Try to calculate indicators
        result_data = calculate_indicators_with_ui_feedback(invalid_data)

        # Assert: Should return original data unchanged
        pd.testing.assert_frame_equal(result_data, invalid_data)

    def test_calculate_indicators_with_ui_feedback_empty_data(self, mock_streamlit):
        """Test UI feedback function with empty data."""
        # Arrange: Empty DataFrame
        empty_data = pd.DataFrame()

        # Act: Calculate indicators
        result_data = calculate_indicators_with_ui_feedback(empty_data)

        # Assert: Should return empty DataFrame
        assert result_data.empty, "Empty input should return empty DataFrame"


class TestIndicatorsMathematicalAccuracy:
    """Test mathematical accuracy of indicator calculations."""

    def test_sma_calculation_against_manual_calculation(self):
        """Test SMA calculation against manually calculated values."""
        # Arrange: Create precise test data with known SMA values
        test_prices = pd.Series(
            [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0]
        )
        window = 3

        # Act: Calculate SMA
        result = TechnicalIndicators.calculate_sma(test_prices, window)

        # Assert: Verify against manual calculations
        # First two values should be NaN
        assert pd.isna(result.iloc[0]), "First value should be NaN"
        assert pd.isna(result.iloc[1]), "Second value should be NaN"

        # Manual calculations
        expected_values = [
            (100.0 + 102.0 + 104.0) / 3,  # 102.0
            (102.0 + 104.0 + 106.0) / 3,  # 104.0
            (104.0 + 106.0 + 108.0) / 3,  # 106.0
            (106.0 + 108.0 + 110.0) / 3,  # 108.0
            (108.0 + 110.0 + 112.0) / 3,  # 110.0
            (110.0 + 112.0 + 114.0) / 3,  # 112.0
        ]

        for i, expected in enumerate(expected_values):
            actual = result.iloc[i + 2]  # Skip first two NaN values
            assert (
                abs(actual - expected) < 1e-10
            ), f"SMA calculation error at position {i+2}"

    def test_sma_with_same_values(self):
        """Test SMA calculation when all values are the same."""
        # Arrange: All same values
        same_values = pd.Series([100.0] * 10)
        window = 5

        # Act: Calculate SMA
        result = TechnicalIndicators.calculate_sma(same_values, window)

        # Assert: SMA should be 100.0 for all calculated positions
        for i in range(window - 1, len(same_values)):
            assert (
                abs(result.iloc[i] - 100.0) < 1e-10
            ), f"SMA should be 100.0 at position {i}"

    def test_sma_precision_with_floating_point(self):
        """Test SMA calculation maintains precision with floating point numbers."""
        # Arrange: Prices with decimal places
        precise_prices = pd.Series([100.123, 102.456, 104.789, 106.012, 108.345])
        window = 3

        # Act: Calculate SMA
        result = TechnicalIndicators.calculate_sma(precise_prices, window)

        # Assert: Manual calculation for verification
        expected_sma_3 = (100.123 + 102.456 + 104.789) / 3
        assert (
            abs(result.iloc[2] - expected_sma_3) < 1e-10
        ), "Precision should be maintained"

        expected_sma_4 = (102.456 + 104.789 + 106.012) / 3
        assert (
            abs(result.iloc[3] - expected_sma_4) < 1e-10
        ), "Precision should be maintained"
