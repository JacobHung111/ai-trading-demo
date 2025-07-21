"""
Tests for Configuration Management Module

This module tests the configuration management functionality including
parameter validation, environment variable loading, and configuration
management across different application contexts.

Author: AI Trading Demo Team
Version: 1.0 (Hybrid Architecture)
"""

import pytest
import datetime
import os
from unittest.mock import patch
from shared.config import (
    TradingConfig,
    get_config,
    update_config,
    reset_config,
    get_streamlit_config,
    get_nicegui_config,
    load_config_from_env,
)


class TestTradingConfig:
    """Test class for TradingConfig functionality."""

    def test_trading_config_initialization_defaults(self):
        """Test TradingConfig initialization with default values."""
        # Arrange & Act: Create TradingConfig with defaults
        config = TradingConfig()

        # Assert: Verify all default values
        assert config.sma_short_period == 20, "Default short SMA period should be 20"
        assert config.sma_long_period == 50, "Default long SMA period should be 50"
        assert (
            config.default_cache_duration == 300
        ), "Default cache duration should be 5 minutes"
        assert (
            config.realtime_cache_duration == 60
        ), "Real-time cache duration should be 1 minute"
        assert config.default_ticker == "AAPL", "Default ticker should be AAPL"
        assert config.default_period_days == 90, "Default period should be 90 days"
        assert config.streamlit_port == 8501, "Default Streamlit port should be 8501"
        assert config.nicegui_port == 8080, "Default NiceGUI port should be 8080"
        assert config.min_data_points == 60, "Minimum data points should be 60"
        assert config.max_date_range_years == 20, "Max date range should be 20 years"
        assert (
            config.realtime_update_interval == 30
        ), "Real-time update interval should be 30 seconds"
        assert config.max_concurrent_updates == 5, "Max concurrent updates should be 5"
        assert config.chart_height == 400, "Chart height should be 400"
        assert (
            config.chart_theme == "plotly_white"
        ), "Chart theme should be plotly_white"
        assert config.signal_sensitivity == 0.001, "Signal sensitivity should be 0.001"

    def test_trading_config_initialization_custom_values(self):
        """Test TradingConfig initialization with custom values."""
        # Arrange & Act: Create TradingConfig with custom values
        config = TradingConfig(
            sma_short_period=10,
            sma_long_period=30,
            default_ticker="MSFT",
            default_period_days=180,
        )

        # Assert: Verify custom values are set
        assert config.sma_short_period == 10, "Custom short SMA period should be 10"
        assert config.sma_long_period == 30, "Custom long SMA period should be 30"
        assert config.default_ticker == "MSFT", "Custom ticker should be MSFT"
        assert config.default_period_days == 180, "Custom period should be 180 days"

        # Verify defaults are still used for unspecified values
        assert (
            config.default_cache_duration == 300
        ), "Unspecified values should use defaults"

    def test_get_default_date_range(self):
        """Test getting default date range."""
        # Arrange: TradingConfig with specific period
        config = TradingConfig(default_period_days=60)

        # Act: Get default date range
        date_range = config.get_default_date_range()

        # Assert: Verify date range structure and values
        assert "start_date" in date_range, "Should include start_date"
        assert "end_date" in date_range, "Should include end_date"
        assert isinstance(
            date_range["start_date"], datetime.date
        ), "start_date should be date object"
        assert isinstance(
            date_range["end_date"], datetime.date
        ), "end_date should be date object"

        # Verify date range calculation
        today = datetime.date.today()
        expected_start = today - datetime.timedelta(days=60)
        assert date_range["end_date"] == today, "End date should be today"
        assert (
            date_range["start_date"] == expected_start
        ), "Start date should be 60 days ago"

    def test_get_cache_duration_streamlit(self):
        """Test getting cache duration for Streamlit application."""
        # Arrange: TradingConfig
        config = TradingConfig()

        # Act: Get cache duration for Streamlit
        cache_duration = config.get_cache_duration("streamlit")

        # Assert: Should return default cache duration
        assert (
            cache_duration == config.default_cache_duration
        ), "Should return default cache duration for Streamlit"

    def test_get_cache_duration_nicegui(self):
        """Test getting cache duration for NiceGUI application."""
        # Arrange: TradingConfig
        config = TradingConfig()

        # Act: Get cache duration for NiceGUI
        cache_duration = config.get_cache_duration("nicegui")

        # Assert: Should return real-time cache duration
        assert (
            cache_duration == config.realtime_cache_duration
        ), "Should return real-time cache duration for NiceGUI"

    def test_get_cache_duration_default(self):
        """Test getting cache duration with default app type."""
        # Arrange: TradingConfig
        config = TradingConfig()

        # Act: Get cache duration with no app type specified
        cache_duration = config.get_cache_duration()

        # Assert: Should return default cache duration
        assert (
            cache_duration == config.default_cache_duration
        ), "Should return default cache duration when no app type specified"

    def test_get_cache_duration_case_insensitive(self):
        """Test cache duration lookup is case insensitive."""
        # Arrange: TradingConfig
        config = TradingConfig()

        # Act: Test various case combinations
        nicegui_lower = config.get_cache_duration("nicegui")
        nicegui_upper = config.get_cache_duration("NICEGUI")
        nicegui_mixed = config.get_cache_duration("NiceGUI")

        # Assert: All should return the same value
        assert (
            nicegui_lower
            == nicegui_upper
            == nicegui_mixed
            == config.realtime_cache_duration
        )

    def test_validate_sma_periods_valid(self):
        """Test SMA period validation with valid configurations."""
        # Arrange: TradingConfig with valid SMA periods
        config = TradingConfig(sma_short_period=20, sma_long_period=50)

        # Act: Validate SMA periods
        is_valid = config.validate_sma_periods()

        # Assert: Should be valid
        assert is_valid is True, "Valid SMA configuration should pass validation"

    def test_validate_sma_periods_invalid_short_greater_than_long(self):
        """Test SMA period validation when short > long."""
        # Arrange: TradingConfig with invalid SMA periods (short > long)
        config = TradingConfig(sma_short_period=50, sma_long_period=20)

        # Act: Validate SMA periods
        is_valid = config.validate_sma_periods()

        # Assert: Should be invalid
        assert (
            is_valid is False
        ), "Short period greater than long period should be invalid"

    def test_validate_sma_periods_invalid_zero_values(self):
        """Test SMA period validation with zero values."""
        # Arrange: TradingConfig with zero SMA periods
        config = TradingConfig(sma_short_period=0, sma_long_period=50)

        # Act: Validate SMA periods
        is_valid = config.validate_sma_periods()

        # Assert: Should be invalid
        assert is_valid is False, "Zero SMA period should be invalid"

    def test_validate_sma_periods_invalid_long_exceeds_min_data(self):
        """Test SMA period validation when long period exceeds minimum data points."""
        # Arrange: TradingConfig with long period exceeding min data points
        config = TradingConfig(
            sma_short_period=20, sma_long_period=70, min_data_points=60
        )

        # Act: Validate SMA periods
        is_valid = config.validate_sma_periods()

        # Assert: Should be invalid
        assert (
            is_valid is False
        ), "Long period exceeding min data points should be invalid"

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        # Arrange: TradingConfig with specific values
        config = TradingConfig(sma_short_period=15, default_ticker="TSLA")

        # Act: Convert to dictionary
        config_dict = config.to_dict()

        # Assert: Verify dictionary structure and values
        assert isinstance(config_dict, dict), "Should return dictionary"
        assert (
            config_dict["sma_short_period"] == 15
        ), "Should include custom short period"
        assert config_dict["default_ticker"] == "TSLA", "Should include custom ticker"
        assert (
            config_dict["sma_long_period"] == 50
        ), "Should include default long period"

        # Verify all expected keys are present
        expected_keys = [
            "sma_short_period",
            "sma_long_period",
            "default_cache_duration",
            "realtime_cache_duration",
            "default_ticker",
            "default_period_days",
            "streamlit_port",
            "nicegui_port",
            "min_data_points",
            "max_date_range_years",
            "realtime_update_interval",
            "max_concurrent_updates",
            "chart_height",
            "chart_theme",
            "signal_sensitivity",
        ]
        for key in expected_keys:
            assert key in config_dict, f"Dictionary should include key: {key}"


class TestGlobalConfigManagement:
    """Test global configuration management functions."""

    def test_get_config_returns_global_instance(self):
        """Test that get_config returns the global configuration instance."""
        # Arrange & Act: Get global config
        config = get_config()

        # Assert: Should return TradingConfig instance
        assert isinstance(config, TradingConfig), "Should return TradingConfig instance"

        # Verify it's the same instance on multiple calls
        config2 = get_config()
        assert config is config2, "Should return same instance on multiple calls"

    def test_update_config_valid_parameters(self):
        """Test updating configuration with valid parameters."""
        # Arrange: Reset config and get original values
        reset_config()
        original_ticker = get_config().default_ticker
        original_period = get_config().sma_short_period

        # Act: Update configuration
        update_config(default_ticker="GOOGL", sma_short_period=15)

        # Assert: Configuration should be updated
        updated_config = get_config()
        assert updated_config.default_ticker == "GOOGL", "Ticker should be updated"
        assert updated_config.sma_short_period == 15, "Short period should be updated"

        # Cleanup: Reset for other tests
        reset_config()

    def test_update_config_invalid_parameter_name(self):
        """Test updating configuration with invalid parameter name."""
        # Arrange: Reset config
        reset_config()

        # Act & Assert: Should raise ValueError for invalid parameter
        with pytest.raises(ValueError, match="Unknown configuration parameter"):
            update_config(invalid_parameter="value")

    def test_update_config_invalid_sma_validation(self):
        """Test updating configuration with invalid SMA periods."""
        # Arrange: Reset config
        reset_config()

        # Act & Assert: Should raise ValueError for invalid SMA configuration
        with pytest.raises(ValueError, match="Invalid SMA period configuration"):
            update_config(sma_short_period=60, sma_long_period=20)  # short > long

    def test_reset_config(self):
        """Test resetting configuration to defaults."""
        # Arrange: Modify configuration
        update_config(default_ticker="MODIFIED")
        assert get_config().default_ticker == "MODIFIED", "Config should be modified"

        # Act: Reset configuration
        reset_config()

        # Assert: Configuration should return to defaults
        assert get_config().default_ticker == "AAPL", "Should reset to default ticker"
        assert isinstance(
            get_config(), TradingConfig
        ), "Should still be TradingConfig instance"


class TestApplicationSpecificConfigurations:
    """Test application-specific configuration functions."""

    def test_get_streamlit_config(self):
        """Test getting Streamlit-specific configuration."""
        # Arrange: Reset config
        reset_config()

        # Act: Get Streamlit config
        streamlit_config = get_streamlit_config()

        # Assert: Verify Streamlit-specific values
        assert isinstance(streamlit_config, dict), "Should return dictionary"
        assert (
            streamlit_config["cache_duration"] == 300
        ), "Should use default cache duration"
        assert (
            streamlit_config["update_interval"] is None
        ), "Should have no automatic updates"

        # Verify all base config values are included
        base_config = get_config().to_dict()
        for key, value in base_config.items():
            assert (
                streamlit_config[key] == value
            ), f"Should include base config value for {key}"

    def test_get_nicegui_config(self):
        """Test getting NiceGUI-specific configuration."""
        # Arrange: Reset config
        reset_config()

        # Act: Get NiceGUI config
        nicegui_config = get_nicegui_config()

        # Assert: Verify NiceGUI-specific values
        assert isinstance(nicegui_config, dict), "Should return dictionary"
        assert (
            nicegui_config["cache_duration"] == 60
        ), "Should use real-time cache duration"
        assert (
            nicegui_config["update_interval"] == 30
        ), "Should use real-time update interval"

        # Verify all base config values are included
        base_config = get_config().to_dict()
        for key, value in base_config.items():
            assert (
                nicegui_config[key] == value
            ), f"Should include base config value for {key}"


class TestEnvironmentVariableLoading:
    """Test loading configuration from environment variables."""

    def test_load_config_from_env_no_variables(self):
        """Test loading config when no environment variables are set."""
        # Arrange: Reset config and ensure no relevant env vars
        reset_config()
        original_ticker = get_config().default_ticker

        # Act: Load from environment (no variables set)
        load_config_from_env()

        # Assert: Configuration should remain unchanged
        assert (
            get_config().default_ticker == original_ticker
        ), "Config should not change without env vars"

    @patch.dict(os.environ, {"CONFIG_DEFAULT_TICKER": "NVDA"})
    def test_load_config_from_env_string_value(self):
        """Test loading string configuration from environment."""
        # Arrange: Reset config
        reset_config()

        # Act: Load from environment
        load_config_from_env()

        # Assert: Configuration should be updated from environment
        assert (
            get_config().default_ticker == "NVDA"
        ), "Should load ticker from environment"

        # Cleanup
        reset_config()

    @patch.dict(os.environ, {"CONFIG_SMA_SHORT": "25"})
    def test_load_config_from_env_integer_value(self):
        """Test loading integer configuration from environment."""
        # Arrange: Reset config
        reset_config()

        # Act: Load from environment
        load_config_from_env()

        # Assert: Configuration should be updated from environment
        assert (
            get_config().sma_short_period == 25
        ), "Should load integer from environment"

        # Cleanup
        reset_config()

    @patch.dict(os.environ, {"CONFIG_SMA_SHORT": "invalid_number"})
    def test_load_config_from_env_invalid_type_conversion(self):
        """Test handling invalid type conversion from environment."""
        # Arrange: Reset config and get original value
        reset_config()
        original_value = get_config().sma_short_period

        # Act: Load from environment (invalid integer)
        load_config_from_env()

        # Assert: Configuration should remain unchanged
        assert (
            get_config().sma_short_period == original_value
        ), "Should not change with invalid conversion"

    @patch.dict(
        os.environ,
        {
            "CONFIG_DEFAULT_TICKER": "AMZN",
            "CONFIG_SMA_SHORT": "30",
            "CONFIG_CACHE_DURATION": "600",
        },
    )
    def test_load_config_from_env_multiple_variables(self):
        """Test loading multiple configuration values from environment."""
        # Arrange: Reset config
        reset_config()

        # Act: Load from environment
        load_config_from_env()

        # Assert: All values should be updated
        config = get_config()
        assert config.default_ticker == "AMZN", "Should load ticker from environment"
        assert (
            config.sma_short_period == 30
        ), "Should load short period from environment"
        assert (
            config.default_cache_duration == 600
        ), "Should load cache duration from environment"

        # Cleanup
        reset_config()

    @patch.dict(os.environ, {"CONFIG_STREAMLIT_PORT": "8502"})
    def test_load_config_from_env_port_configuration(self):
        """Test loading port configuration from environment."""
        # Arrange: Reset config
        reset_config()

        # Act: Load from environment
        load_config_from_env()

        # Assert: Port should be updated
        assert get_config().streamlit_port == 8502, "Should load port from environment"

        # Cleanup
        reset_config()


class TestConfigurationEdgeCases:
    """Test edge cases and error scenarios for configuration."""

    def test_config_with_extreme_values(self):
        """Test configuration with extreme but valid values."""
        # Arrange & Act: Create config with extreme values
        config = TradingConfig(
            sma_short_period=1,
            sma_long_period=2,
            min_data_points=2,
            default_period_days=1,
            max_date_range_years=1,
            chart_height=100,
            signal_sensitivity=0.0001,
        )

        # Assert: Should handle extreme values correctly
        assert (
            config.validate_sma_periods() is True
        ), "Extreme but valid values should pass validation"
        assert config.sma_short_period == 1, "Should accept minimum short period"
        assert config.sma_long_period == 2, "Should accept minimum long period"

    def test_config_date_range_boundary_conditions(self):
        """Test date range calculation with boundary conditions."""
        # Arrange: Config with 1-day period
        config = TradingConfig(default_period_days=1)

        # Act: Get date range
        date_range = config.get_default_date_range()

        # Assert: Should handle single-day range correctly
        expected_start = datetime.date.today() - datetime.timedelta(days=1)
        assert (
            date_range["start_date"] == expected_start
        ), "Should handle 1-day period correctly"
        assert (
            date_range["end_date"] == datetime.date.today()
        ), "End date should be today"

    def test_config_to_dict_completeness(self):
        """Test that to_dict includes all configuration parameters."""
        # Arrange: Create config
        config = TradingConfig()

        # Act: Convert to dictionary
        config_dict = config.to_dict()

        # Assert: Should include all attributes
        expected_attributes = [
            attr
            for attr in dir(config)
            if not attr.startswith("_") and not callable(getattr(config, attr))
        ]

        for attr in expected_attributes:
            assert attr in config_dict, f"Dictionary should include attribute: {attr}"

    def test_concurrent_config_updates(self):
        """Test thread safety of configuration updates."""
        # Arrange: Reset config
        reset_config()
        original_ticker = get_config().default_ticker

        # Act: Multiple rapid updates
        update_config(default_ticker="TEST1")
        update_config(default_ticker="TEST2")
        update_config(default_ticker="TEST3")

        # Assert: Final update should be applied
        assert get_config().default_ticker == "TEST3", "Final update should be applied"

        # Cleanup
        reset_config()

    def test_config_parameter_type_consistency(self):
        """Test that configuration parameters maintain type consistency."""
        # Arrange: Create config with specific types
        config = TradingConfig()

        # Act & Assert: Verify types are maintained
        assert isinstance(
            config.sma_short_period, int
        ), "SMA periods should be integers"
        assert isinstance(config.sma_long_period, int), "SMA periods should be integers"
        assert isinstance(config.default_ticker, str), "Ticker should be string"
        assert isinstance(
            config.signal_sensitivity, float
        ), "Sensitivity should be float"
        assert isinstance(config.chart_height, int), "Chart height should be integer"
        assert isinstance(config.chart_theme, str), "Chart theme should be string"
