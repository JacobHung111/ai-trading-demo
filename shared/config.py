"""
Configuration Management for AI Trading Demo

Centralized configuration management providing consistent parameters
across both Streamlit and NiceGUI applications.

Author: AI Trading Demo Team
Version: 1.0 (Hybrid Architecture)
"""

import datetime
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class TradingConfig:
    """Trading strategy configuration parameters."""

    # Moving Average Periods
    sma_short_period: int = 20
    sma_long_period: int = 50

    # Data Management
    default_cache_duration: int = 300  # 5 minutes for Streamlit
    realtime_cache_duration: int = 60  # 1 minute for NiceGUI

    # Default Ticker and Date Range
    default_ticker: str = "AAPL"
    default_period_days: int = 90

    # Application Settings
    streamlit_port: int = 8501
    nicegui_port: int = 8080

    # Data Validation
    min_data_points: int = 60  # Minimum for 50-day SMA
    max_date_range_years: int = 20

    # Real-time Update Settings
    realtime_update_interval: int = 30  # seconds
    max_concurrent_updates: int = 5

    # Chart Configuration
    chart_height: int = 400
    chart_theme: str = "plotly_white"

    # Signal Detection
    signal_sensitivity: float = 0.001  # Minimum price difference for crossover

    def get_default_date_range(self) -> Dict[str, datetime.date]:
        """Get default date range for analysis.

        Returns:
            Dict containing start_date and end_date.
        """
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=self.default_period_days)
        return {"start_date": start_date, "end_date": end_date}

    def get_cache_duration(self, app_type: str = "streamlit") -> int:
        """Get appropriate cache duration for application type.

        Args:
            app_type (str): Application type ("streamlit" or "nicegui").

        Returns:
            int: Cache duration in seconds.
        """
        if app_type.lower() == "nicegui":
            return self.realtime_cache_duration
        return self.default_cache_duration

    def validate_sma_periods(self) -> bool:
        """Validate SMA period configuration.

        Returns:
            bool: True if configuration is valid.
        """
        return (
            self.sma_short_period > 0
            and self.sma_long_period > 0
            and self.sma_short_period < self.sma_long_period
            and self.sma_long_period <= self.min_data_points
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dict containing all configuration parameters.
        """
        return {
            "sma_short_period": self.sma_short_period,
            "sma_long_period": self.sma_long_period,
            "default_cache_duration": self.default_cache_duration,
            "realtime_cache_duration": self.realtime_cache_duration,
            "default_ticker": self.default_ticker,
            "default_period_days": self.default_period_days,
            "streamlit_port": self.streamlit_port,
            "nicegui_port": self.nicegui_port,
            "min_data_points": self.min_data_points,
            "max_date_range_years": self.max_date_range_years,
            "realtime_update_interval": self.realtime_update_interval,
            "max_concurrent_updates": self.max_concurrent_updates,
            "chart_height": self.chart_height,
            "chart_theme": self.chart_theme,
            "signal_sensitivity": self.signal_sensitivity,
        }


# Global configuration instance
CONFIG = TradingConfig()


def get_config() -> TradingConfig:
    """Get the global configuration instance.

    Returns:
        TradingConfig: The global configuration object.
    """
    return CONFIG


def update_config(**kwargs) -> None:
    """Update configuration parameters.

    Args:
        **kwargs: Configuration parameters to update.

    Raises:
        ValueError: If invalid configuration provided.
    """
    global CONFIG

    for key, value in kwargs.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

    if not CONFIG.validate_sma_periods():
        raise ValueError("Invalid SMA period configuration")


def reset_config() -> None:
    """Reset configuration to default values."""
    global CONFIG
    CONFIG = TradingConfig()


# Application-specific configuration helpers
def get_streamlit_config() -> Dict[str, Any]:
    """Get configuration optimized for Streamlit application.

    Returns:
        Dict: Streamlit-specific configuration.
    """
    config = CONFIG.to_dict()
    config["cache_duration"] = CONFIG.default_cache_duration
    config["update_interval"] = None  # Manual updates
    return config


def get_nicegui_config() -> Dict[str, Any]:
    """Get configuration optimized for NiceGUI application.

    Returns:
        Dict: NiceGUI-specific configuration.
    """
    config = CONFIG.to_dict()
    config["cache_duration"] = CONFIG.realtime_cache_duration
    config["update_interval"] = CONFIG.realtime_update_interval
    return config


# Environment-specific overrides (for future use)
def load_config_from_env() -> None:
    """Load configuration overrides from environment variables.

    Looks for environment variables with CONFIG_ prefix and updates
    the global configuration accordingly.
    """
    import os

    env_mappings = {
        "CONFIG_SMA_SHORT": ("sma_short_period", int),
        "CONFIG_SMA_LONG": ("sma_long_period", int),
        "CONFIG_CACHE_DURATION": ("default_cache_duration", int),
        "CONFIG_DEFAULT_TICKER": ("default_ticker", str),
        "CONFIG_STREAMLIT_PORT": ("streamlit_port", int),
        "CONFIG_NICEGUI_PORT": ("nicegui_port", int),
    }

    updates = {}
    for env_var, (config_key, type_func) in env_mappings.items():
        if env_var in os.environ:
            try:
                updates[config_key] = type_func(os.environ[env_var])
            except ValueError:
                print(f"Warning: Invalid value for {env_var}, using default")

    if updates:
        update_config(**updates)
