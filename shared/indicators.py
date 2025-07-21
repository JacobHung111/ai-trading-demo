"""
Technical Indicators Module for AI Trading Demo

This module provides technical indicator calculations used across both
Streamlit and NiceGUI applications. Currently implements Simple Moving
Averages (SMA) with extensibility for additional indicators.

Author: AI Trading Demo Team
Version: 1.0 (Hybrid Architecture)
"""

import pandas as pd
from typing import Optional


class TechnicalIndicators:
    """Technical indicators calculation class with optimization for real-time updates."""

    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average for a given window period.

        This function computes the SMA using pandas rolling window functionality,
        optimized for both historical analysis and real-time updates.

        Args:
            data (pd.Series): Price series data (typically Close prices).
            window (int): The number of periods for the moving average.

        Returns:
            pd.Series: Simple Moving Average values with same index as input.
                      Initial periods will contain NaN values.
        """
        if len(data) < window:
            # Return series of NaN if insufficient data
            return pd.Series([float("nan")] * len(data), index=data.index)

        return data.rolling(window=window, min_periods=window).mean()

    @staticmethod
    def calculate_sma20(data: pd.Series) -> pd.Series:
        """Calculate 20-day Simple Moving Average.

        Convenience method for the short-term SMA used in our trading strategy.

        Args:
            data (pd.Series): Price series data (typically Close prices).

        Returns:
            pd.Series: 20-day SMA values.
        """
        return TechnicalIndicators.calculate_sma(data, 20)

    @staticmethod
    def calculate_sma50(data: pd.Series) -> pd.Series:
        """Calculate 50-day Simple Moving Average.

        Convenience method for the long-term SMA used in our trading strategy.

        Args:
            data (pd.Series): Price series data (typically Close prices).

        Returns:
            pd.Series: 50-day SMA values.
        """
        return TechnicalIndicators.calculate_sma(data, 50)

    @staticmethod
    def add_all_indicators(
        data: pd.DataFrame, price_column: str = "Close"
    ) -> pd.DataFrame:
        """Add all technical indicators to the DataFrame.

        This function adds SMA20 and SMA50 columns to the input DataFrame,
        preserving the original data and adding new indicator columns.

        Args:
            data (pd.DataFrame): OHLCV stock data.
            price_column (str): Name of the price column to use for calculations.
                               Defaults to 'Close'.

        Returns:
            pd.DataFrame: Original data with added SMA20 and SMA50 columns.
                         Returns empty DataFrame if calculation fails.
        """
        try:
            # Create a copy to avoid modifying original data
            result_data = data.copy()

            # Validate that the price column exists
            if price_column not in result_data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")

            # Calculate indicators
            result_data["SMA20"] = TechnicalIndicators.calculate_sma20(
                result_data[price_column]
            )
            result_data["SMA50"] = TechnicalIndicators.calculate_sma50(
                result_data[price_column]
            )

            return result_data

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_latest_indicators(
        data: pd.DataFrame, price_column: str = "Close"
    ) -> Optional[dict]:
        """Get the latest indicator values for real-time display.

        This function returns the most recent indicator values, useful for
        real-time applications that need current SMA values.

        Args:
            data (pd.DataFrame): OHLCV stock data with indicators calculated.
            price_column (str): Name of the price column. Defaults to 'Close'.

        Returns:
            Optional[dict]: Dictionary containing latest values:
                           {'price': float, 'sma20': float, 'sma50': float}
                           Returns None if data is insufficient or invalid.
        """
        try:
            if data.empty or len(data) < 50:
                return None

            # Get the latest row with valid SMA50 data
            valid_data = data.dropna(subset=["SMA20", "SMA50"])
            if valid_data.empty:
                return None

            latest = valid_data.iloc[-1]

            return {
                "price": float(latest[price_column]),
                "sma20": float(latest["SMA20"]),
                "sma50": float(latest["SMA50"]),
                "date": latest.get("Date", latest.name),
            }

        except Exception as e:
            print(f"Error getting latest indicators: {e}")
            return None

    @staticmethod
    def validate_indicator_data(data: pd.DataFrame) -> bool:
        """Validate that indicator data is sufficient for signal generation.

        This function checks if the DataFrame contains enough valid indicator
        data to generate reliable trading signals.

        Args:
            data (pd.DataFrame): DataFrame with calculated indicators.

        Returns:
            bool: True if data is valid for signal generation, False otherwise.
        """
        try:
            # Check if required columns exist
            required_columns = ["SMA20", "SMA50"]
            if not all(col in data.columns for col in required_columns):
                return False

            # Check if we have sufficient non-NaN data
            valid_data = data.dropna(subset=required_columns)

            # Need at least 2 rows for crossover detection
            return len(valid_data) >= 2

        except Exception:
            return False


# Additional utility functions moved from utils.py
def calculate_indicators_with_ui_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Simple Moving Averages (SMA) with UI feedback.

    This function adds SMA20 and SMA50 columns to the input DataFrame and
    removes rows with NaN values that result from the moving average calculations.
    Provides Streamlit UI feedback for better user experience.

    Args:
        df (pd.DataFrame): Input DataFrame containing stock price data with
                          at least a 'Close' column.

    Returns:
        pd.DataFrame: DataFrame with added SMA20 and SMA50 columns, with
                      initial NaN rows removed. Returns the original DataFrame
                      if the 'Close' column is missing.
    """
    try:
        # Validate input DataFrame has required column
        if df.empty or "Close" not in df.columns:
            try:
                import streamlit as st

                st.warning(
                    "DataFrame is empty or missing 'Close' column for indicator calculation."
                )
            except ImportError:
                print(
                    "Warning: DataFrame is empty or missing 'Close' column for indicator calculation."
                )
            return df

        # Use the main TechnicalIndicators class for calculation
        return TechnicalIndicators.add_all_indicators(df)

    except Exception as e:
        try:
            import streamlit as st

            st.error(f"Error calculating indicators: {str(e)}")
        except ImportError:
            print(f"Error calculating indicators: {str(e)}")
        return df
