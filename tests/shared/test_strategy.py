"""
Tests for Trading Strategy Module

This module tests the logical correctness of the SMA crossover trading strategy
including signal generation, crossover detection, and all edge cases following
the precise specifications in claude.md Chapter 7.2.

Author: AI Trading Demo Team
Version: 1.0 (Hybrid Architecture)
"""

import pytest
import pandas as pd
import numpy as np
import datetime
from shared.strategy import (
    TradingStrategy,
    generate_trading_signals_with_ui_feedback,
    get_signal_summary_with_ui_feedback,
)


class TestTradingStrategy:
    """Test class for TradingStrategy functionality."""

    def test_strategy_initialization(self):
        """Test TradingStrategy initialization with default and custom parameters."""
        # Arrange & Act: Create strategy with default parameters
        default_strategy = TradingStrategy()

        # Assert: Verify default parameters
        assert default_strategy.short_window == 20, "Default short window should be 20"
        assert default_strategy.long_window == 50, "Default long window should be 50"

        # Arrange & Act: Create strategy with custom parameters
        custom_strategy = TradingStrategy(short_window=10, long_window=30)

        # Assert: Verify custom parameters
        assert custom_strategy.short_window == 10, "Custom short window should be 10"
        assert custom_strategy.long_window == 30, "Custom long window should be 30"

    def test_generate_signals_buy_crossover(self, crossover_scenario_data):
        """Test buy signal generation when SMA20 crosses above SMA50.

        Per Chapter 7.2: Buy Signal occurs when SMA20 crosses above SMA50
        Precise Logic: (SMA20 of previous day < SMA50 of previous day) AND
                      (SMA20 of current day > SMA50 of current day)
        """
        # Arrange: Use crossover scenario data that has a designed buy signal at index 25
        strategy = TradingStrategy()
        test_data = crossover_scenario_data.copy()

        # Act: Generate signals
        result_data = strategy.generate_signals(test_data)

        # Assert: Verify buy signal is generated correctly
        assert "Signal" in result_data.columns, "Signal column should be added"

        # Verify the buy signal at the crossover point (index 25)
        buy_signal_row = result_data.iloc[25]
        assert (
            buy_signal_row["Signal"] == 1
        ), "Should generate buy signal at crossover point"

        # Verify crossover conditions are met
        previous_row = result_data.iloc[24]
        assert (
            previous_row["SMA20"] < previous_row["SMA50"]
        ), "Previous day: SMA20 should be below SMA50"
        assert (
            buy_signal_row["SMA20"] > buy_signal_row["SMA50"]
        ), "Current day: SMA20 should be above SMA50"

        # Verify no buy signals in other positions around the crossover
        assert result_data.iloc[24]["Signal"] == 0, "No signal before crossover"
        assert (
            result_data.iloc[26]["Signal"] == 0
        ), "No signal after crossover (unless another crossover)"

    def test_generate_signals_sell_crossover(self, crossover_scenario_data):
        """Test sell signal generation when SMA20 crosses below SMA50.

        Per Chapter 7.2: Sell Signal occurs when SMA20 crosses below SMA50
        Precise Logic: (SMA20 of previous day > SMA50 of previous day) AND
                      (SMA20 of current day < SMA50 of current day)
        """
        # Arrange: Use crossover scenario data that has a designed sell signal at index 40
        strategy = TradingStrategy()
        test_data = crossover_scenario_data.copy()

        # Act: Generate signals
        result_data = strategy.generate_signals(test_data)

        # Assert: Verify sell signal is generated correctly
        sell_signal_row = result_data.iloc[40]
        assert (
            sell_signal_row["Signal"] == -1
        ), "Should generate sell signal at crossover point"

        # Verify crossover conditions are met
        previous_row = result_data.iloc[39]
        assert (
            previous_row["SMA20"] > previous_row["SMA50"]
        ), "Previous day: SMA20 should be above SMA50"
        assert (
            sell_signal_row["SMA20"] < sell_signal_row["SMA50"]
        ), "Current day: SMA20 should be below SMA50"

        # Verify no sell signals in other positions around the crossover
        assert result_data.iloc[39]["Signal"] == 0, "No signal before crossover"
        assert (
            result_data.iloc[41]["Signal"] == 0
        ), "No signal after crossover (unless another crossover)"

    def test_generate_signals_no_signal_consistently_above(self):
        """Test no signal when SMA20 is consistently above SMA50."""
        # Arrange: Create data where SMA20 is always above SMA50
        test_data = pd.DataFrame(
            {
                "Date": [
                    datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                    for i in range(10)
                ],
                "Close": [100.0 + i for i in range(10)],
                "SMA20": [105.0 + i for i in range(10)],  # Always above SMA50
                "SMA50": [100.0 + i for i in range(10)],
            }
        )
        strategy = TradingStrategy()

        # Act: Generate signals
        result_data = strategy.generate_signals(test_data)

        # Assert: No signals should be generated
        assert all(
            result_data["Signal"] == 0
        ), "No signals when SMA20 consistently above SMA50"

    def test_generate_signals_no_signal_consistently_below(self):
        """Test no signal when SMA20 is consistently below SMA50."""
        # Arrange: Create data where SMA20 is always below SMA50
        test_data = pd.DataFrame(
            {
                "Date": [
                    datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                    for i in range(10)
                ],
                "Close": [100.0 + i for i in range(10)],
                "SMA20": [95.0 + i for i in range(10)],  # Always below SMA50
                "SMA50": [100.0 + i for i in range(10)],
            }
        )
        strategy = TradingStrategy()

        # Act: Generate signals
        result_data = strategy.generate_signals(test_data)

        # Assert: No signals should be generated
        assert all(
            result_data["Signal"] == 0
        ), "No signals when SMA20 consistently below SMA50"

    def test_generate_signals_no_signal_equal_values(self):
        """Test no signal when SMA20 equals SMA50."""
        # Arrange: Create data where SMA20 equals SMA50
        test_data = pd.DataFrame(
            {
                "Date": [
                    datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                    for i in range(10)
                ],
                "Close": [100.0 + i for i in range(10)],
                "SMA20": [100.0 + i for i in range(10)],  # Equal to SMA50
                "SMA50": [100.0 + i for i in range(10)],
            }
        )
        strategy = TradingStrategy()

        # Act: Generate signals
        result_data = strategy.generate_signals(test_data)

        # Assert: No signals should be generated
        assert all(result_data["Signal"] == 0), "No signals when SMA20 equals SMA50"

    def test_generate_signals_missing_columns(self):
        """Test signal generation with missing required columns."""
        # Arrange: Data without SMA columns
        invalid_data = pd.DataFrame(
            {"Date": [datetime.date(2023, 1, 1)], "Close": [100.0]}
        )
        strategy = TradingStrategy()

        # Act: Try to generate signals
        result_data = strategy.generate_signals(invalid_data)

        # Assert: Should return empty DataFrame due to error
        assert (
            result_data.empty
        ), "Should return empty DataFrame when required columns missing"

    def test_generate_signals_empty_data(self):
        """Test signal generation with empty DataFrame."""
        # Arrange: Empty DataFrame
        empty_data = pd.DataFrame()
        strategy = TradingStrategy()

        # Act: Try to generate signals
        result_data = strategy.generate_signals(empty_data)

        # Assert: Should return empty DataFrame
        assert result_data.empty, "Should return empty DataFrame for empty input"

    def test_get_latest_signal_buy_signal(self, crossover_scenario_data):
        """Test getting the latest buy signal."""
        # Arrange: Generate signals and get data with buy signal
        strategy = TradingStrategy()
        data_with_signals = strategy.generate_signals(crossover_scenario_data)

        # Act: Get latest signal
        latest_signal = strategy.get_latest_signal(data_with_signals)

        # Assert: Should return the most recent signal (sell signal at index 40)
        assert latest_signal is not None, "Should return a signal"
        assert latest_signal["signal"] == -1, "Latest signal should be sell signal"
        assert latest_signal["type"] == "SELL", "Signal type should be SELL"
        assert "date" in latest_signal, "Should include date"
        assert "price" in latest_signal, "Should include price"
        assert "sma20" in latest_signal, "Should include SMA20"
        assert "sma50" in latest_signal, "Should include SMA50"

    def test_get_latest_signal_no_signals(self):
        """Test getting latest signal when no signals exist."""
        # Arrange: Data with no signals (SMA20 consistently above SMA50)
        no_signal_data = pd.DataFrame(
            {
                "Date": [
                    datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                    for i in range(5)
                ],
                "Close": [100.0 + i for i in range(5)],
                "SMA20": [105.0 + i for i in range(5)],
                "SMA50": [100.0 + i for i in range(5)],
                "Signal": [0, 0, 0, 0, 0],
            }
        )
        strategy = TradingStrategy()

        # Act: Try to get latest signal
        latest_signal = strategy.get_latest_signal(no_signal_data)

        # Assert: Should return None
        assert latest_signal is None, "Should return None when no signals exist"

    def test_get_latest_signal_empty_data(self):
        """Test getting latest signal from empty data."""
        # Arrange: Empty DataFrame
        empty_data = pd.DataFrame()
        strategy = TradingStrategy()

        # Act: Try to get latest signal
        latest_signal = strategy.get_latest_signal(empty_data)

        # Assert: Should return None
        assert latest_signal is None, "Should return None for empty data"

    def test_get_signal_summary(self, crossover_scenario_data):
        """Test generating signal summary statistics."""
        # Arrange: Generate signals
        strategy = TradingStrategy()
        data_with_signals = strategy.generate_signals(crossover_scenario_data)

        # Act: Get signal summary
        summary = strategy.get_signal_summary(data_with_signals)

        # Assert: Verify summary structure and content
        assert summary is not None, "Should return summary"
        assert "total_signals" in summary, "Should include total signals count"
        assert "buy_signals" in summary, "Should include buy signals count"
        assert "sell_signals" in summary, "Should include sell signals count"
        assert "total_days" in summary, "Should include total days"
        assert "signal_rate" in summary, "Should include signal rate"

        # Verify expected signal counts (based on crossover_scenario_data design)
        assert summary["buy_signals"] == 1, "Should have 1 buy signal"
        assert summary["sell_signals"] == 1, "Should have 1 sell signal"
        assert summary["total_signals"] == 2, "Should have 2 total signals"
        assert summary["total_days"] == len(
            data_with_signals
        ), "Total days should match data length"

        # Verify signal rate calculation
        expected_rate = (2 / len(data_with_signals)) * 100
        assert (
            abs(summary["signal_rate"] - expected_rate) < 0.01
        ), "Signal rate calculation should be correct"

    def test_get_signal_summary_no_signals(self):
        """Test signal summary with no signals."""
        # Arrange: Data with no signals
        no_signal_data = pd.DataFrame({"Signal": [0, 0, 0, 0, 0]})
        strategy = TradingStrategy()

        # Act: Get signal summary
        summary = strategy.get_signal_summary(no_signal_data)

        # Assert: Verify summary with zero signals
        assert summary["total_signals"] == 0, "Should have 0 total signals"
        assert summary["buy_signals"] == 0, "Should have 0 buy signals"
        assert summary["sell_signals"] == 0, "Should have 0 sell signals"
        assert summary["signal_rate"] == 0.0, "Signal rate should be 0.0"

    def test_get_signal_list(self, crossover_scenario_data):
        """Test getting list of all signals."""
        # Arrange: Generate signals
        strategy = TradingStrategy()
        data_with_signals = strategy.generate_signals(crossover_scenario_data)

        # Act: Get signal list
        signal_list = strategy.get_signal_list(data_with_signals)

        # Assert: Verify signal list structure and content
        assert isinstance(signal_list, list), "Should return a list"
        assert len(signal_list) == 2, "Should have 2 signals (1 buy, 1 sell)"

        # Verify first signal (buy signal)
        first_signal = signal_list[0]
        assert first_signal["signal"] == 1, "First signal should be buy (1)"
        assert first_signal["type"] == "BUY", "First signal type should be BUY"
        assert "date" in first_signal, "Should include date"
        assert "price" in first_signal, "Should include price"
        assert "sma20" in first_signal, "Should include SMA20"
        assert "sma50" in first_signal, "Should include SMA50"
        assert "volume" in first_signal, "Should include volume"

        # Verify second signal (sell signal)
        second_signal = signal_list[1]
        assert second_signal["signal"] == -1, "Second signal should be sell (-1)"
        assert second_signal["type"] == "SELL", "Second signal type should be SELL"

    def test_check_crossover_conditions_buy_crossover(self):
        """Test crossover detection for buy signals."""
        # Arrange: Set up crossover scenario data
        previous_data = pd.Series({"SMA20": 99.0, "SMA50": 100.0})  # SMA20 below SMA50
        current_data = pd.Series({"SMA20": 101.0, "SMA50": 100.0})  # SMA20 above SMA50
        strategy = TradingStrategy()

        # Act: Check crossover conditions
        signal = strategy.check_crossover_conditions(current_data, previous_data)

        # Assert: Should detect buy signal
        assert signal == 1, "Should detect buy signal when SMA20 crosses above SMA50"

    def test_check_crossover_conditions_sell_crossover(self):
        """Test crossover detection for sell signals."""
        # Arrange: Set up crossover scenario data
        previous_data = pd.Series({"SMA20": 101.0, "SMA50": 100.0})  # SMA20 above SMA50
        current_data = pd.Series({"SMA20": 99.0, "SMA50": 100.0})  # SMA20 below SMA50
        strategy = TradingStrategy()

        # Act: Check crossover conditions
        signal = strategy.check_crossover_conditions(current_data, previous_data)

        # Assert: Should detect sell signal
        assert signal == -1, "Should detect sell signal when SMA20 crosses below SMA50"

    def test_check_crossover_conditions_no_crossover(self):
        """Test crossover detection when no crossover occurs."""
        # Arrange: No crossover scenario
        previous_data = pd.Series({"SMA20": 101.0, "SMA50": 100.0})  # SMA20 above SMA50
        current_data = pd.Series(
            {"SMA20": 102.0, "SMA50": 100.0}
        )  # Still SMA20 above SMA50
        strategy = TradingStrategy()

        # Act: Check crossover conditions
        signal = strategy.check_crossover_conditions(current_data, previous_data)

        # Assert: Should detect no signal
        assert signal == 0, "Should detect no signal when no crossover occurs"

    def test_check_crossover_conditions_missing_data(self):
        """Test crossover detection with missing data."""
        # Arrange: Missing required fields
        previous_data = pd.Series({"SMA20": 101.0})  # Missing SMA50
        current_data = pd.Series({"SMA20": 102.0, "SMA50": 100.0})
        strategy = TradingStrategy()

        # Act: Check crossover conditions
        signal = strategy.check_crossover_conditions(current_data, previous_data)

        # Assert: Should return None due to insufficient data
        assert signal is None, "Should return None when required data is missing"

    def test_get_strategy_parameters(self):
        """Test getting strategy parameters."""
        # Arrange: Create strategy with specific parameters
        strategy = TradingStrategy(short_window=15, long_window=40)

        # Act: Get parameters
        params = strategy.get_strategy_parameters()

        # Assert: Verify parameter structure and values
        assert params["short_window"] == 15, "Should return correct short window"
        assert params["long_window"] == 40, "Should return correct long window"
        assert params["strategy_name"] == "SMA Crossover", "Should return strategy name"
        assert params["version"] == "2.0", "Should return version"


class TestStrategyUIFeedbackFunctions:
    """Test UI feedback functions for Streamlit integration."""

    def test_generate_trading_signals_with_ui_feedback(
        self, sample_with_indicators, mock_streamlit
    ):
        """Test signal generation with UI feedback using valid data."""
        # Arrange: Use sample data with indicators
        input_data = sample_with_indicators.copy()

        # Act: Generate signals with UI feedback
        result_data = generate_trading_signals_with_ui_feedback(input_data)

        # Assert: Should return data with signals
        assert "Signal" in result_data.columns, "Signal column should be added"
        assert len(result_data) == len(input_data), "Row count should be preserved"

        # Verify signals are properly calculated
        signal_counts = result_data["Signal"].value_counts()
        assert 0 in signal_counts, "Should have no-signal periods"

    def test_generate_trading_signals_with_ui_feedback_missing_columns(
        self, mock_streamlit
    ):
        """Test UI feedback function with missing required columns."""
        # Arrange: Data without SMA columns
        invalid_data = pd.DataFrame({"Close": [100.0, 102.0, 104.0]})

        # Act: Try to generate signals
        result_data = generate_trading_signals_with_ui_feedback(invalid_data)

        # Assert: Should return original data unchanged
        pd.testing.assert_frame_equal(result_data, invalid_data)

    def test_get_signal_summary_with_ui_feedback(
        self, crossover_scenario_data, mock_streamlit
    ):
        """Test signal summary with UI feedback."""
        # Arrange: Generate signals first
        strategy = TradingStrategy()
        data_with_signals = strategy.generate_signals(crossover_scenario_data)

        # Act: Get signal summary with UI feedback
        summary = get_signal_summary_with_ui_feedback(data_with_signals)

        # Assert: Should return valid summary
        assert "total_signals" in summary, "Should include total signals"
        assert "buy_signals" in summary, "Should include buy signals"
        assert "sell_signals" in summary, "Should include sell signals"
        assert "no_signal_days" in summary, "Should include no signal days"
        assert "total_days" in summary, "Should include total days"

    def test_get_signal_summary_with_ui_feedback_missing_signal_column(
        self, mock_streamlit
    ):
        """Test UI feedback summary function with missing Signal column."""
        # Arrange: Data without Signal column
        invalid_data = pd.DataFrame({"Close": [100.0, 102.0, 104.0]})

        # Act: Try to get summary
        summary = get_signal_summary_with_ui_feedback(invalid_data)

        # Assert: Should return empty dictionary
        assert summary == {}, "Should return empty dict when Signal column missing"


class TestStrategyPrecisionAndEdgeCases:
    """Test precision and edge cases for trading strategy logic."""

    def test_crossover_detection_precision(self):
        """Test crossover detection with very small price differences."""
        # Arrange: Create data with minimal crossover differences
        test_data = pd.DataFrame(
            {
                "Date": [
                    datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                    for i in range(3)
                ],
                "Close": [100.0, 100.0, 100.0],
                "SMA20": [100.001, 99.999, 100.001],  # Crosses below then above
                "SMA50": [100.000, 100.000, 100.000],
            }
        )
        strategy = TradingStrategy()

        # Act: Generate signals
        result_data = strategy.generate_signals(test_data)

        # Assert: Should detect crossover even with small differences
        assert (
            result_data.iloc[1]["Signal"] == -1
        ), "Should detect sell crossover with small difference"
        assert (
            result_data.iloc[2]["Signal"] == 1
        ), "Should detect buy crossover with small difference"

    def test_multiple_consecutive_crossovers(self):
        """Test handling of multiple consecutive crossovers."""
        # Arrange: Data with rapid crossovers
        test_data = pd.DataFrame(
            {
                "Date": [
                    datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                    for i in range(6)
                ],
                "Close": [100.0] * 6,
                "SMA20": [
                    99.0,
                    101.0,
                    99.0,
                    101.0,
                    99.0,
                    101.0,
                ],  # Alternating crossovers
                "SMA50": [100.0] * 6,
            }
        )
        strategy = TradingStrategy()

        # Act: Generate signals
        result_data = strategy.generate_signals(test_data)

        # Assert: Should generate signals for each crossover
        assert (
            result_data.iloc[1]["Signal"] == 1
        ), "Should generate buy signal at first crossover"
        assert (
            result_data.iloc[2]["Signal"] == -1
        ), "Should generate sell signal at second crossover"
        assert (
            result_data.iloc[3]["Signal"] == 1
        ), "Should generate buy signal at third crossover"
        assert (
            result_data.iloc[4]["Signal"] == -1
        ), "Should generate sell signal at fourth crossover"
        assert (
            result_data.iloc[5]["Signal"] == 1
        ), "Should generate buy signal at fifth crossover"

    def test_signal_generation_with_nan_values(self):
        """Test signal generation when data contains NaN values."""
        # Arrange: Data with NaN values in SMA columns
        test_data = pd.DataFrame(
            {
                "Date": [
                    datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                    for i in range(5)
                ],
                "Close": [100.0, 102.0, 104.0, 106.0, 108.0],
                "SMA20": [np.nan, np.nan, 99.0, 101.0, 103.0],
                "SMA50": [np.nan, np.nan, 100.0, 100.0, 100.0],
            }
        )
        strategy = TradingStrategy()

        # Act: Generate signals
        result_data = strategy.generate_signals(test_data)

        # Assert: Should handle NaN values gracefully
        assert "Signal" in result_data.columns, "Should add Signal column"
        assert len(result_data) == len(test_data), "Should preserve all rows"

        # Signals should only be possible where both SMAs are available
        # and there's a previous row for comparison
        assert result_data.iloc[0]["Signal"] == 0, "No signal possible with NaN"
        assert result_data.iloc[1]["Signal"] == 0, "No signal possible with NaN"
        assert (
            result_data.iloc[2]["Signal"] == 0
        ), "No signal possible without previous non-NaN"
        # Only from index 3 onwards can we have valid signals
        assert (
            result_data.iloc[3]["Signal"] == 1
        ), "Should detect buy crossover at valid position"
