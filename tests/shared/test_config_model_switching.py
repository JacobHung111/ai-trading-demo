"""
Comprehensive Test Suite for AI Model Switching Functionality

This module tests the AI model switching functionality implemented in shared/config.py,
including model configuration updates, rate limit adjustments, error handling,
API response differences, and cache invalidation.

Author: AI Trading Demo Team
Version: 2.0 (AI-Powered Architecture)
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from core.config import AITradingConfig, get_config, reset_config


class TestAIModelSwitching:
    """Test suite for AI model switching functionality."""

    def test_update_model_settings_valid_model(self, ai_trading_config: AITradingConfig):
        """Test updating model settings with a valid hardcoded model.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange
        initial_model = ai_trading_config.ai_model_name
        initial_requests = ai_trading_config.gemini_rate_limit_requests
        initial_tokens = ai_trading_config.gemini_rate_limit_tokens
        initial_max_tokens = ai_trading_config.ai_max_tokens
        
        target_model = "gemini-1.5-pro"
        
        # Act
        result = ai_trading_config.update_model_settings(target_model)
        
        # Assert
        assert result is True
        assert ai_trading_config.ai_model_name == target_model
        assert ai_trading_config.gemini_rate_limit_requests == 2  # Pro model limit
        assert ai_trading_config.gemini_rate_limit_tokens == 32000
        assert ai_trading_config.ai_max_tokens == 4096
        
        # Verify changes from initial values
        assert ai_trading_config.ai_model_name != initial_model
        assert ai_trading_config.gemini_rate_limit_requests != initial_requests
        assert ai_trading_config.ai_max_tokens != initial_max_tokens

    def test_update_model_settings_invalid_model(self, ai_trading_config: AITradingConfig):
        """Test updating model settings with an invalid model name.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange
        initial_model = ai_trading_config.ai_model_name
        initial_requests = ai_trading_config.gemini_rate_limit_requests
        invalid_model = "non-existent-model"
        
        # Act
        result = ai_trading_config.update_model_settings(invalid_model)
        
        # Assert
        assert result is False
        assert ai_trading_config.ai_model_name == initial_model  # Unchanged
        assert ai_trading_config.gemini_rate_limit_requests == initial_requests  # Unchanged

    @patch('core.config.AITradingConfig.fetch_models_from_api')
    def test_rate_limits_updated_correctly_for_different_models(self, mock_fetch, ai_trading_config: AITradingConfig):
        """Test that rate limits are correctly updated for different model types.
        
        Args:
            mock_fetch: Mocked fetch method to ensure predictable hardcoded values.
            ai_trading_config: Fresh config instance from fixture.
        """
        # Force use of hardcoded models for predictable testing
        mock_fetch.return_value = None
        
        # Test Flash Lite model - check hardcoded configured values
        ai_trading_config.update_model_settings("gemini-2.0-flash-lite")
        assert ai_trading_config.gemini_rate_limit_requests == 12  # Hardcoded value
        assert ai_trading_config.gemini_rate_limit_tokens == 32000
        assert ai_trading_config.ai_max_tokens == 1000
        
        # Test Flash model  
        ai_trading_config.update_model_settings("gemini-1.5-flash")
        assert ai_trading_config.gemini_rate_limit_requests == 15  # Hardcoded value
        assert ai_trading_config.gemini_rate_limit_tokens == 1000000
        assert ai_trading_config.ai_max_tokens == 2048
        
        # Test Pro model
        ai_trading_config.update_model_settings("gemini-1.5-pro")
        assert ai_trading_config.gemini_rate_limit_requests == 2  # Hardcoded value
        assert ai_trading_config.gemini_rate_limit_tokens == 32000
        assert ai_trading_config.ai_max_tokens == 4096

    @patch('google.genai')
    def test_fetch_models_from_api_success(self, mock_genai, ai_trading_config: AITradingConfig):
        """Test successful dynamic model fetching from Google Gemini API.
        
        Args:
            mock_genai: Mocked Google GenAI client.
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange
        ai_trading_config.google_api_key = "test_api_key"
        
        # Create mock models response
        mock_model_1 = Mock()
        mock_model_1.name = "models/gemini-1.5-pro"
        mock_model_1.display_name = "Gemini 1.5 Pro"
        mock_model_1.description = "Most capable model"
        
        mock_model_2 = Mock()
        mock_model_2.name = "models/gemini-1.5-flash"
        mock_model_2.display_name = "Gemini 1.5 Flash"
        mock_model_2.description = "Fast model"
        
        mock_client = Mock()
        mock_client.models.list.return_value = [mock_model_1, mock_model_2]
        mock_genai.Client.return_value = mock_client
        
        # Act
        result = ai_trading_config.fetch_models_from_api()
        
        # Assert
        assert result is not None
        assert len(result) == 2
        assert "gemini-1.5-pro" in result
        assert "gemini-1.5-flash" in result
        
        # Verify model details
        pro_model = result["gemini-1.5-pro"]
        assert pro_model["name"] == "Gemini 1.5 Pro"
        assert pro_model["description"] == "Most capable model"
        assert pro_model["rate_limit_requests"] == 2  # Pro model
        assert pro_model["source"] == "api"
        assert pro_model["api_name"] == "models/gemini-1.5-pro"
        
        flash_model = result["gemini-1.5-flash"]
        assert flash_model["name"] == "Gemini 1.5 Flash"
        assert flash_model["rate_limit_requests"] == 15  # Flash model

    @patch('google.genai')
    def test_fetch_models_from_api_no_api_key(self, mock_genai, ai_trading_config: AITradingConfig):
        """Test model fetching fails gracefully when no API key is available.
        
        Args:
            mock_genai: Mocked Google GenAI client.
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange
        ai_trading_config.google_api_key = None
        
        # Act
        result = ai_trading_config.fetch_models_from_api()
        
        # Assert
        assert result is None
        mock_genai.Client.assert_not_called()

    @patch('google.genai')
    @patch('google.api_core.exceptions')
    def test_fetch_models_from_api_google_api_error(self, mock_exceptions, mock_genai, ai_trading_config: AITradingConfig):
        """Test model fetching handles Google API errors gracefully.
        
        Args:
            mock_exceptions: Mocked Google API exceptions.
            mock_genai: Mocked Google GenAI client.
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange
        ai_trading_config.google_api_key = "test_api_key"
        
        mock_client = Mock()
        mock_client.models.list.side_effect = mock_exceptions.GoogleAPICallError("API Error")
        mock_genai.Client.return_value = mock_client
        
        # Act
        result = ai_trading_config.fetch_models_from_api()
        
        # Assert
        assert result is None

    @patch('google.genai')
    def test_fetch_models_from_api_general_error(self, mock_genai, ai_trading_config: AITradingConfig):
        """Test model fetching handles general exceptions gracefully.
        
        Args:
            mock_genai: Mocked Google GenAI client.
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange
        ai_trading_config.google_api_key = "test_api_key"
        
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("Network Error")
        mock_genai.Client.return_value = mock_client
        
        # Act
        result = ai_trading_config.fetch_models_from_api()
        
        # Assert
        assert result is None

    def test_get_available_models_fallback_to_hardcoded(self, ai_trading_config: AITradingConfig):
        """Test that get_available_models falls back to hardcoded models when API fails.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange - No API key to ensure fallback
        ai_trading_config.google_api_key = None
        
        # Act
        result = ai_trading_config.get_available_models()
        
        # Assert
        assert result is not None
        assert len(result) == 3  # Hardcoded models
        assert "gemini-2.0-flash-lite" in result
        assert "gemini-1.5-flash" in result
        assert "gemini-1.5-pro" in result
        
        # Verify it returns copies (not references to original)
        assert result is not ai_trading_config.available_ai_models

    @patch('core.config.AITradingConfig.fetch_models_from_api')
    def test_get_available_models_with_dynamic_models(self, mock_fetch, ai_trading_config: AITradingConfig):
        """Test that get_available_models returns dynamic models when available.
        
        Args:
            mock_fetch: Mocked fetch_models_from_api method.
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange
        dynamic_models = {
            "gemini-2.0-pro": {
                "name": "Gemini 2.0 Pro",
                "rate_limit_requests": 5,
                "rate_limit_tokens": 50000,
                "max_tokens": 8192,
                "source": "api"
            }
        }
        mock_fetch.return_value = dynamic_models
        
        # Act
        result = ai_trading_config.get_available_models()
        
        # Assert
        assert result == dynamic_models
        mock_fetch.assert_called_once()

    def test_get_current_model_info_valid_model(self, ai_trading_config: AITradingConfig):
        """Test getting current model information for a valid model.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange
        ai_trading_config.update_model_settings("gemini-1.5-pro")
        
        # Act
        result = ai_trading_config.get_current_model_info()
        
        # Assert
        assert result["current_model"] == "gemini-1.5-pro"
        assert result["name"] == "Gemini 1.5 Pro"
        assert result["current_rate_limit_requests"] == 2
        assert result["current_rate_limit_tokens"] == 32000
        assert result["current_max_tokens"] == 4096

    def test_get_current_model_info_invalid_model(self, ai_trading_config: AITradingConfig):
        """Test getting current model information for an invalid model.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange - Force an invalid model name (bypass validation)
        ai_trading_config.ai_model_name = "invalid-model"
        
        # Act
        result = ai_trading_config.get_current_model_info()
        
        # Assert
        assert result["current_model"] == "invalid-model"
        assert result["name"] == "Unknown Model"
        assert result["description"] == "Model configuration not found"

    @patch('core.config.AITradingConfig.fetch_models_from_api')  
    def test_update_model_settings_with_dynamic_models(self, mock_fetch, ai_trading_config: AITradingConfig):
        """Test updating model settings when dynamic models are available.
        
        Args:
            mock_fetch: Mocked fetch_models_from_api method.
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange
        dynamic_models = {
            "gemini-2.0-pro": {
                "name": "Gemini 2.0 Pro",
                "rate_limit_requests": 5,
                "rate_limit_tokens": 50000,
                "max_tokens": 8192,
                "description": "Latest Pro model",
                "tier": "free",
                "source": "api"
            }
        }
        mock_fetch.return_value = dynamic_models
        
        # Act
        result = ai_trading_config.update_model_settings("gemini-2.0-pro")
        
        # Assert
        assert result is True
        assert ai_trading_config.ai_model_name == "gemini-2.0-pro"
        assert ai_trading_config.gemini_rate_limit_requests == 5
        assert ai_trading_config.gemini_rate_limit_tokens == 50000
        assert ai_trading_config.ai_max_tokens == 8192

    def test_estimate_rate_limits_from_model_name(self, ai_trading_config: AITradingConfig):
        """Test rate limit estimation from model name patterns.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Test Pro model detection
        pro_limits = ai_trading_config._estimate_rate_limits_from_model_name("gemini-1.5-pro")
        assert pro_limits["requests"] == 2
        assert pro_limits["tokens"] == 32000
        assert pro_limits["max_tokens"] == 4096
        
        # Test Flash Lite model detection
        flash_lite_limits = ai_trading_config._estimate_rate_limits_from_model_name("gemini-2.0-flash-lite")
        assert flash_lite_limits["requests"] == 15
        assert flash_lite_limits["tokens"] == 32000
        assert flash_lite_limits["max_tokens"] == 1000
        
        # Test Regular Flash model detection
        flash_limits = ai_trading_config._estimate_rate_limits_from_model_name("gemini-1.5-flash")
        assert flash_limits["requests"] == 15
        assert flash_limits["tokens"] == 1000000
        assert flash_limits["max_tokens"] == 2048
        
        # Test Unknown model (fallback)
        unknown_limits = ai_trading_config._estimate_rate_limits_from_model_name("unknown-model")
        assert unknown_limits["requests"] == 12
        assert unknown_limits["tokens"] == 32000
        assert unknown_limits["max_tokens"] == 1000

    @patch('core.config.AITradingConfig.fetch_models_from_api')
    def test_refresh_available_models_success(self, mock_fetch, ai_trading_config: AITradingConfig):
        """Test successful refresh of available models.
        
        Args:
            mock_fetch: Mocked fetch_models_from_api method.
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange
        new_models = {
            "gemini-3.0-beta": {
                "name": "Gemini 3.0 Beta",
                "rate_limit_requests": 1,
                "rate_limit_tokens": 10000,
                "max_tokens": 16384,
                "source": "api"
            }
        }
        mock_fetch.return_value = new_models
        
        initial_count = len(ai_trading_config.available_ai_models)
        
        # Act
        result = ai_trading_config.refresh_available_models()
        
        # Assert
        assert result is True
        assert len(ai_trading_config.available_ai_models) > initial_count
        assert "gemini-3.0-beta" in ai_trading_config.available_ai_models

    @patch('core.config.AITradingConfig.fetch_models_from_api')
    def test_refresh_available_models_failure(self, mock_fetch, ai_trading_config: AITradingConfig):
        """Test refresh handling when API fails.
        
        Args:
            mock_fetch: Mocked fetch_models_from_api method.
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange
        mock_fetch.return_value = None
        initial_models = ai_trading_config.available_ai_models.copy()
        
        # Act
        result = ai_trading_config.refresh_available_models()
        
        # Assert
        assert result is False
        assert ai_trading_config.available_ai_models == initial_models  # Unchanged

    def test_detect_rate_limits_dynamically_fallback(self, ai_trading_config: AITradingConfig):
        """Test dynamic rate limit detection falls back to configured defaults.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Test with known model
        result = ai_trading_config.detect_rate_limits_dynamically("gemini-1.5-pro")
        assert result["requests_per_minute"] == 2
        assert result["tokens_per_minute"] == 32000
        
        # Test with unknown model
        result = ai_trading_config.detect_rate_limits_dynamically("unknown-model")
        assert result["requests_per_minute"] == 12  # Default fallback
        assert result["tokens_per_minute"] == 32000

    def test_model_switching_configuration_consistency(self, ai_trading_config: AITradingConfig):
        """Test that model switching maintains configuration consistency.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Test switching between multiple models
        models_to_test = ["gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"]
        
        for model in models_to_test:
            # Act
            success = ai_trading_config.update_model_settings(model)
            
            # Assert
            assert success is True
            assert ai_trading_config.ai_model_name == model
            
            # Verify rate limits are consistent with model specs
            model_info = ai_trading_config.get_current_model_info()
            assert model_info["current_rate_limit_requests"] > 0
            assert model_info["current_rate_limit_tokens"] > 0
            assert model_info["current_max_tokens"] > 0
            
            # Verify configuration validation still passes
            assert ai_trading_config.validate_rate_limits() is True

    def test_model_switching_logging(self, ai_trading_config: AITradingConfig, caplog):
        """Test that model switching generates appropriate log messages.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
            caplog: Pytest log capture fixture.
        """
        # Test successful model switch
        with caplog.at_level(logging.INFO):
            ai_trading_config.update_model_settings("gemini-1.5-pro")
        
        # Check for expected log messages
        log_messages = [record.message for record in caplog.records]
        assert any("Updated configuration for model: Gemini 1.5 Pro" in msg for msg in log_messages)
        assert any("Rate limits: 2 req/min, 32000 tokens/min" in msg for msg in log_messages)
        
        # Test invalid model switch
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            ai_trading_config.update_model_settings("invalid-model")
        
        log_messages = [record.message for record in caplog.records]
        assert any("Unknown AI model: invalid-model" in msg for msg in log_messages)

    def test_concurrent_model_operations(self, ai_trading_config: AITradingConfig):
        """Test that concurrent model operations handle state correctly.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Simulate rapid model switches
        models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"]
        
        for model in models:
            ai_trading_config.update_model_settings(model)
            current_info = ai_trading_config.get_current_model_info()
            
            # Verify state consistency
            assert current_info["current_model"] == model
            assert ai_trading_config.ai_model_name == model
            
            # Verify available models don't change during switching
            available = ai_trading_config.get_available_models()
            assert len(available) >= 3  # At least the hardcoded models

    @patch('core.config.AITradingConfig.fetch_models_from_api')
    def test_api_response_differences_between_models(self, mock_fetch, ai_trading_config: AITradingConfig):
        """Test that different models have different API response characteristics.
        
        Args:
            mock_fetch: Mocked fetch_models_from_api method.
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange - Mock different models with varying characteristics
        dynamic_models = {
            "gemini-lite": {
                "name": "Gemini Lite",
                "rate_limit_requests": 60,  # High request rate
                "rate_limit_tokens": 10000,  # Low token rate
                "max_tokens": 512,
                "description": "Fast, lightweight model",
                "tier": "free",
                "source": "api"
            },
            "gemini-ultra": {
                "name": "Gemini Ultra",
                "rate_limit_requests": 1,  # Low request rate
                "rate_limit_tokens": 100000,  # High token rate
                "max_tokens": 32768,
                "description": "Most capable but rate-limited model",
                "tier": "premium",
                "source": "api"
            }
        }
        mock_fetch.return_value = dynamic_models
        
        # Test Lite model characteristics
        ai_trading_config.update_model_settings("gemini-lite")
        assert ai_trading_config.gemini_rate_limit_requests == 60
        assert ai_trading_config.gemini_rate_limit_tokens == 10000
        assert ai_trading_config.ai_max_tokens == 512
        
        # Test Ultra model characteristics  
        ai_trading_config.update_model_settings("gemini-ultra")
        assert ai_trading_config.gemini_rate_limit_requests == 1
        assert ai_trading_config.gemini_rate_limit_tokens == 100000
        assert ai_trading_config.ai_max_tokens == 32768
        
        # Verify significant differences
        lite_info = ai_trading_config.get_available_models()["gemini-lite"]
        ultra_info = ai_trading_config.get_available_models()["gemini-ultra"]
        
        assert lite_info["rate_limit_requests"] != ultra_info["rate_limit_requests"]
        assert lite_info["max_tokens"] != ultra_info["max_tokens"]
        assert lite_info["tier"] != ultra_info["tier"]

    def test_cache_invalidation_on_model_switch(self, ai_trading_config: AITradingConfig):
        """Test that cache-related settings remain consistent during model switches.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Arrange - Record initial cache settings
        initial_ai_cache = ai_trading_config.ai_cache_duration
        initial_news_cache = ai_trading_config.news_cache_duration
        initial_default_cache = ai_trading_config.default_cache_duration
        
        # Act - Switch models multiple times
        models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"]
        for model in models:
            ai_trading_config.update_model_settings(model)
            
            # Assert - Cache durations should remain unchanged
            assert ai_trading_config.ai_cache_duration == initial_ai_cache
            assert ai_trading_config.news_cache_duration == initial_news_cache
            assert ai_trading_config.default_cache_duration == initial_default_cache
            
            # Cache duration getters should still work
            assert ai_trading_config.get_cache_duration("ai") == initial_ai_cache
            assert ai_trading_config.get_cache_duration("news") == initial_news_cache

    def test_dynamic_vs_hardcoded_model_behavior(self, ai_trading_config: AITradingConfig):
        """Test that dynamic models override hardcoded models when available.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Test with dynamic models available (real API call scenario)
        if ai_trading_config.google_api_key:
            # First, try to get dynamic models
            dynamic_models = ai_trading_config.fetch_models_from_api()
            
            if dynamic_models and "gemini-2.0-flash-lite" in dynamic_models:
                # Dynamic model values may differ from hardcoded
                ai_trading_config.update_model_settings("gemini-2.0-flash-lite")
                dynamic_rate_limit = dynamic_models["gemini-2.0-flash-lite"]["rate_limit_requests"]
                assert ai_trading_config.gemini_rate_limit_requests == dynamic_rate_limit
                
                # Verify the model info reflects dynamic source
                model_info = ai_trading_config.get_current_model_info()
                available_models = ai_trading_config.get_available_models()
                if "gemini-2.0-flash-lite" in available_models:
                    assert available_models["gemini-2.0-flash-lite"].get("source") == "api"
            else:
                # Skip test if dynamic models not available
                pytest.skip("Dynamic models not available from API")
        else:
            pytest.skip("No API key available for dynamic model testing")


class TestModelSwitchingIntegration:
    """Integration tests for model switching with other system components."""

    @patch('core.config.AITradingConfig.fetch_models_from_api')
    def test_post_init_model_update(self, mock_fetch):
        """Test that __post_init__ calls update_model_settings correctly."""
        # Force use of hardcoded models for predictable testing
        mock_fetch.return_value = None
        
        # Arrange & Act - Create new config instance (triggers __post_init__)
        config = AITradingConfig()
        
        # Assert - Default model should be set with correct rate limits based on hardcoded configuration
        assert config.ai_model_name == "gemini-2.0-flash-lite"
        # With mocked fetch, should use hardcoded values
        assert config.gemini_rate_limit_requests == 12  # Hardcoded value
        assert config.gemini_rate_limit_tokens == 32000
        assert config.ai_max_tokens == 1000

    @patch('core.config.AITradingConfig.fetch_models_from_api')
    def test_global_config_model_switching(self, mock_fetch):
        """Test model switching with the global configuration instance.
        
        Args:
            mock_fetch: Mocked fetch_models_from_api method.
        """
        # Ensure the mock returns None to use hardcoded models
        mock_fetch.return_value = None
        
        # Reset to ensure clean state and get fresh global config
        reset_config()
        global_config = get_config()
        
        # Verify the global config has access to hardcoded models
        available_models = global_config.get_available_models()
        assert "gemini-1.5-pro" in available_models
        
        # Test switching model on global instance
        success = global_config.update_model_settings("gemini-1.5-pro")
        assert success is True
        assert global_config.ai_model_name == "gemini-1.5-pro"
        
        # Verify global config reflects changes
        current_info = global_config.get_current_model_info()
        assert current_info["current_model"] == "gemini-1.5-pro"

    def test_model_validation_with_config_validation(self, ai_trading_config: AITradingConfig):
        """Test that model switching maintains overall configuration validation.
        
        Args:
            ai_trading_config: Fresh config instance from fixture.
        """
        # Test all hardcoded models maintain valid configuration
        for model_name in ai_trading_config.available_ai_models.keys():
            # Act
            success = ai_trading_config.update_model_settings(model_name)
            
            # Assert
            assert success is True
            assert ai_trading_config.validate_ai_parameters() is True
            assert ai_trading_config.validate_rate_limits() is True
            
            # Verify configuration dictionary is complete
            config_dict = ai_trading_config.to_dict()
            assert "ai_model_name" in config_dict
            assert config_dict["ai_model_name"] == model_name