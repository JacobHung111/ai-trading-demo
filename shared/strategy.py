"""
Trading Strategy Module for AI Trading Demo

This module implements the core SMA crossover trading strategy logic
shared between Streamlit and NiceGUI applications. The strategy generates
buy/sell signals based on 20-day and 50-day Simple Moving Average crossovers.

Author: AI Trading Demo Team
Version: 1.0 (Hybrid Architecture)
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from .indicators import TechnicalIndicators


class TradingStrategy:
    """SMA Crossover trading strategy implementation."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        """Initialize the trading strategy with SMA parameters.

        Args:
            short_window (int): Period for short-term SMA. Defaults to 20.
            long_window (int): Period for long-term SMA. Defaults to 50.
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on SMA crossover strategy.

        This function implements the core trading logic:
        - Buy Signal (1): When SMA20 crosses above SMA50
        - Sell Signal (-1): When SMA20 crosses below SMA50
        - No Signal (0): All other cases

        Args:
            data (pd.DataFrame): OHLCV data with SMA20 and SMA50 calculated.

        Returns:
            pd.DataFrame: Original data with added 'Signal' column.
                         Returns empty DataFrame if signal generation fails.
        """
        try:
            # Create a copy to avoid modifying original data
            result_data = data.copy()

            # Validate that required columns exist
            required_columns = ["SMA20", "SMA50"]
            if not all(col in result_data.columns for col in required_columns):
                raise ValueError("Data must contain SMA20 and SMA50 columns")

            # Initialize signal column with zeros
            result_data["Signal"] = 0

            # Generate buy signals (SMA20 crosses above SMA50)
            # Per Chapter 7.2: (SMA20 of previous day < SMA50 of previous day) AND (SMA20 of current day > SMA50 of current day)
            buy_condition = (result_data["SMA20"] > result_data["SMA50"]) & (
                result_data["SMA20"].shift(1) < result_data["SMA50"].shift(1)
            )
            result_data.loc[buy_condition, "Signal"] = 1

            # Generate sell signals (SMA20 crosses below SMA50)
            # Per Chapter 7.2: (SMA20 of previous day > SMA50 of previous day) AND (SMA20 of current day < SMA50 of current day)
            sell_condition = (result_data["SMA20"] < result_data["SMA50"]) & (
                result_data["SMA20"].shift(1) > result_data["SMA50"].shift(1)
            )
            result_data.loc[sell_condition, "Signal"] = -1

            return result_data

        except Exception as e:
            print(f"Error generating trading signals: {e}")
            return pd.DataFrame()

    def get_latest_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Get the most recent trading signal and its details.

        This function is useful for real-time applications that need to
        know the current signal status.

        Args:
            data (pd.DataFrame): Data with signals generated.

        Returns:
            Optional[Dict]: Dictionary containing:
                           {'signal': int, 'date': str, 'price': float, 'type': str}
                           Returns None if no valid signal found.
        """
        try:
            if data.empty or "Signal" not in data.columns:
                return None

            # Find the most recent non-zero signal
            signals = data[data["Signal"] != 0]
            if signals.empty:
                return None

            latest_signal = signals.iloc[-1]

            signal_type = "BUY" if latest_signal["Signal"] == 1 else "SELL"

            return {
                "signal": int(latest_signal["Signal"]),
                "date": latest_signal.get("Date", latest_signal.name),
                "price": float(latest_signal["Close"]),
                "type": signal_type,
                "sma20": float(latest_signal["SMA20"]),
                "sma50": float(latest_signal["SMA50"]),
            }

        except Exception as e:
            print(f"Error getting latest signal: {e}")
            return None

    def get_signal_summary(self, data: pd.DataFrame) -> Optional[Dict]:
        """Generate a summary of all trading signals in the dataset.

        This function provides statistical analysis of the strategy performance
        useful for both applications' reporting features.

        Args:
            data (pd.DataFrame): Data with signals generated.

        Returns:
            Optional[Dict]: Dictionary containing signal statistics:
                           {'total_signals', 'buy_signals', 'sell_signals',
                            'total_days', 'signal_rate'}
                           Returns None if calculation fails.
        """
        try:
            if data.empty or "Signal" not in data.columns:
                return None

            total_days = len(data)
            total_signals = len(data[data["Signal"] != 0])
            buy_signals = len(data[data["Signal"] == 1])
            sell_signals = len(data[data["Signal"] == -1])

            signal_rate = (total_signals / total_days * 100) if total_days > 0 else 0

            return {
                "total_signals": total_signals,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "total_days": total_days,
                "signal_rate": round(signal_rate, 2),
            }

        except Exception as e:
            print(f"Error calculating signal summary: {e}")
            return None

    def get_signal_list(self, data: pd.DataFrame) -> List[Dict]:
        """Get a list of all trading signals with their details.

        This function returns all signals in chronological order, useful
        for displaying signal history in both applications.

        Args:
            data (pd.DataFrame): Data with signals generated.

        Returns:
            List[Dict]: List of signal dictionaries, each containing:
                       {'date', 'signal', 'type', 'price', 'sma20', 'sma50'}
                       Returns empty list if no signals found.
        """
        try:
            if data.empty or "Signal" not in data.columns:
                return []

            # Filter to only signal rows
            signals = data[data["Signal"] != 0].copy()
            if signals.empty:
                return []

            signal_list = []
            for _, row in signals.iterrows():
                signal_type = "BUY" if row["Signal"] == 1 else "SELL"

                signal_list.append(
                    {
                        "date": row.get("Date", row.name),
                        "signal": int(row["Signal"]),
                        "type": signal_type,
                        "price": float(row["Close"]),
                        "sma20": float(row["SMA20"]),
                        "sma50": float(row["SMA50"]),
                        "volume": int(row.get("Volume", 0)),
                    }
                )

            return signal_list

        except Exception as e:
            print(f"Error getting signal list: {e}")
            return []

    def check_crossover_conditions(
        self, current_data: pd.Series, previous_data: pd.Series
    ) -> Optional[int]:
        """Check for crossover conditions between two data points.

        This function is optimized for real-time signal detection when
        new data points arrive.

        Args:
            current_data (pd.Series): Current period's data with SMA values.
            previous_data (pd.Series): Previous period's data with SMA values.

        Returns:
            Optional[int]: 1 for buy signal, -1 for sell signal,
                          0 for no signal, None for insufficient data.
        """
        try:
            # Validate required data
            required_fields = ["SMA20", "SMA50"]
            if not all(
                field in current_data and field in previous_data
                for field in required_fields
            ):
                return None

            # Check for buy signal (SMA20 crosses above SMA50)
            # Per Chapter 7.2: strict crossover logic
            if (
                current_data["SMA20"] > current_data["SMA50"]
                and previous_data["SMA20"] < previous_data["SMA50"]
            ):
                return 1

            # Check for sell signal (SMA20 crosses below SMA50)
            # Per Chapter 7.2: strict crossover logic
            if (
                current_data["SMA20"] < current_data["SMA50"]
                and previous_data["SMA20"] > previous_data["SMA50"]
            ):
                return -1

            # No signal
            return 0

        except Exception as e:
            print(f"Error checking crossover conditions: {e}")
            return None

    def get_strategy_parameters(self) -> Dict:
        """Get current strategy parameters.

        Returns:
            Dict: Strategy configuration including window sizes.
        """
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "strategy_name": "SMA Crossover",
            "version": "2.0",
        }


# Additional utility functions moved from utils.py
def generate_trading_signals_with_ui_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """Generates trading signals based on SMA crossover strategy with UI feedback.

    This function implements the precise crossover detection logic specified
    in the project blueprint with Streamlit UI feedback for better user experience.

    Args:
        df (pd.DataFrame): Input DataFrame containing stock data with SMA20
                          and SMA50 columns.

    Returns:
        pd.DataFrame: DataFrame with added 'Signal' column containing:
                      1 for buy signals (SMA20 crosses above SMA50)
                      -1 for sell signals (SMA20 crosses below SMA50)
                      0 for no signal (all other cases)
                      Returns the original DataFrame if required columns are missing.
    """
    try:
        # Validate input DataFrame has required columns
        required_columns = ["SMA20", "SMA50"]
        if df.empty or not all(col in df.columns for col in required_columns):
            try:
                import streamlit as st

                st.warning(
                    "DataFrame is missing required SMA columns for signal generation."
                )
            except ImportError:
                print(
                    "Warning: DataFrame is missing required SMA columns for signal generation."
                )
            return df

        # Use the main TradingStrategy class for signal generation
        strategy = TradingStrategy()
        return strategy.generate_signals(df)

    except Exception as e:
        try:
            import streamlit as st

            st.error(f"Error generating trading signals: {str(e)}")
        except ImportError:
            print(f"Error generating trading signals: {str(e)}")
        return df


def get_signal_summary_with_ui_feedback(df: pd.DataFrame) -> dict:
    """Generates a summary of trading signals with UI feedback.

    This function analyzes the signals in the DataFrame and provides
    a statistical summary with Streamlit UI compatibility.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Signal' column with
                          trading signals (1, -1, 0).

    Returns:
        dict: Dictionary containing signal counts and basic statistics.
              Returns empty dict if Signal column is missing.
    """
    try:
        if df.empty or "Signal" not in df.columns:
            return {}

        signal_counts = df["Signal"].value_counts()

        summary = {
            "total_signals": len(df[df["Signal"] != 0]),
            "buy_signals": signal_counts.get(1, 0),
            "sell_signals": signal_counts.get(-1, 0),
            "no_signal_days": signal_counts.get(0, 0),
            "total_days": len(df),
        }

        return summary

    except Exception as e:
        try:
            import streamlit as st

            st.error(f"Error generating signal summary: {str(e)}")
        except ImportError:
            print(f"Error generating signal summary: {str(e)}")
        return {}
