"""
Configuration Management for AI Trading Demo

Centralized configuration management providing consistent parameters
for the unified AI-powered trading analysis application.

Author: AI Trading Demo Team
Version: 2.0 (AI-Powered Hybrid Architecture)
"""

import datetime
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class AITradingConfig:
    """AI-powered trading strategy configuration parameters."""

    # API Key Management (loaded from environment variables)
    google_api_key: Optional[str] = field(default=None, init=False)
    newsapi_api_key: Optional[str] = field(default=None, init=False)

    # AI Analysis Parameters
    news_articles_count: int = 10  # Number of news articles to analyze
    max_news_age_hours: int = 24  # Maximum age of news articles in hours
    ai_model_name: str = (
        "gemini-2.0-flash-lite"  # Default Gemini model - can be changed via UI
    )
    ai_temperature: float = 0.1  # Low temperature for consistent analysis
    ai_max_tokens: int = 1000  # Maximum tokens for AI response

    # Available AI Models with their characteristics
    available_ai_models: Dict[str, Dict] = field(default_factory=lambda: {
        "gemini-2.0-flash-lite": {
            "name": "Gemini 2.0 Flash (Lite)",
            "rate_limit_requests": 12,  # 12 per minute (conservative for 15/min limit)
            "rate_limit_tokens": 32000,  # Tokens per minute
            "max_tokens": 1000,
            "description": "Fast, efficient model optimized for quick responses",
            "tier": "free"
        },
        "gemini-1.5-flash": {
            "name": "Gemini 1.5 Flash",
            "rate_limit_requests": 15,  # 15 per minute for standard tier
            "rate_limit_tokens": 1000000,  # Higher token limit
            "max_tokens": 2048,
            "description": "Balanced performance and speed",
            "tier": "free"
        },
        "gemini-1.5-pro": {
            "name": "Gemini 1.5 Pro",
            "rate_limit_requests": 2,  # 2 per minute for free tier
            "rate_limit_tokens": 32000,
            "max_tokens": 4096,
            "description": "Most capable model with advanced reasoning",
            "tier": "free"
        }
    })

    # Dynamic Rate Limiting Configuration (based on selected model)
    gemini_rate_limit_requests: int = 12  # Will be updated based on selected model
    gemini_rate_limit_tokens: int = 32000  # Will be updated based on selected model
    newsapi_rate_limit_requests: int = 1000  # Requests per day

    # Retry Configuration (Extended for quota exhaustion)
    max_retries: int = 3  # Maximum retry attempts
    retry_base_delay: float = (
        5.0  # Base delay in seconds for exponential backoff (increased from 1.0)
    )
    retry_max_delay: float = (
        300.0  # Maximum delay between retries (5 minutes for quota recovery)
    )

    # Network Timeout Configuration
    api_timeout_seconds: int = 30  # API request timeout
    connection_timeout_seconds: int = 10  # Connection timeout

    # Data Management
    default_cache_duration: int = 300  # 5 minutes for Streamlit
    news_cache_duration: int = 1800  # 30 minutes for news data
    ai_cache_duration: int = 3600  # 1 hour for AI analysis results
    
    # Model fetching state (not serialized to dict)
    _models_fetched_from_api: bool = field(default=False, init=False)

    # Default Ticker and Date Range (can be overridden by environment variable)
    default_ticker: str = "AAPL"
    default_period_days: int = 90

    # Application Settings
    streamlit_port: int = 8501

    # Data Validation
    min_data_points: int = 30  # Minimum data points for analysis
    max_date_range_years: int = 20

    # Application Settings
    max_concurrent_updates: int = 5

    # Chart Configuration
    chart_height: int = 400
    chart_theme: str = "plotly_white"

    # AI Signal Confidence Thresholds
    min_confidence_threshold: float = 0.6  # Minimum confidence for signal generation
    high_confidence_threshold: float = 0.8  # High confidence threshold

    def __post_init__(self) -> None:
        """Initialize API keys from environment variables after dataclass creation."""
        try:
            self.load_api_keys()
            self.update_model_settings(self.ai_model_name)
        except Exception as e:
            logging.warning(f"Error during config initialization: {e}")
            # Continue with default settings if initialization fails

    def load_api_keys(self) -> None:
        """Load API keys from environment variables or Streamlit secrets.

        Tries to load from Streamlit secrets first, then falls back to environment variables.
        This supports both local development and Streamlit Cloud deployment.
        """
        # Try Streamlit secrets first (for Streamlit Cloud deployment)
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                self.google_api_key = st.secrets.get("GOOGLE_API_KEY")
                self.newsapi_api_key = st.secrets.get("NEWS_API_KEY")  # Note: NEWS_API_KEY in secrets
                if self.google_api_key and self.newsapi_api_key:
                    logging.info("API keys loaded from Streamlit secrets")
                    return
        except (ImportError, AttributeError, KeyError):
            pass
        
        # Fallback to environment variables (for local development)
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.newsapi_api_key = os.getenv("NEWSAPI_API_KEY") or os.getenv("NEWS_API_KEY")

        if not self.google_api_key:
            logging.warning("GOOGLE_API_KEY not found in environment variables or Streamlit secrets")
        if not self.newsapi_api_key:
            logging.warning("NEWS_API_KEY not found in environment variables or Streamlit secrets")

    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are available and not placeholder values.

        Returns:
            Dict[str, bool]: Dictionary indicating availability of each API key.
        """
        google_valid = (
            bool(self.google_api_key)
            and self.google_api_key
            not in ["your_google_gemini_api_key_here", "XXXXXXXXXXXXXXXXXXXXXXX"]
            and len(self.google_api_key) > 20
        )

        newsapi_valid = (
            bool(self.newsapi_api_key)
            and self.newsapi_api_key
            not in ["your_newsapi_key_here", "XXXXXXXXXXXXXXXXXXXXXXX"]
            and len(self.newsapi_api_key) > 10
        )

        return {
            "google_api_key": google_valid,
            "newsapi_api_key": newsapi_valid,
        }

    def get_default_date_range(self) -> Dict[str, datetime.date]:
        """Get default date range for analysis.

        Returns:
            Dict containing start_date and end_date.
        """
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=self.default_period_days)
        return {"start_date": start_date, "end_date": end_date}

    def get_available_models(self) -> Dict[str, Dict]:
        """Get list of available AI models with their characteristics.
        
        Attempts to fetch models dynamically from API on first call, then uses cache.
        
        Returns:
            Dict[str, Dict]: Dictionary of available models.
        """
        # Try to fetch dynamic models on first call (only if we have API key and haven't fetched yet)
        if self.google_api_key and not self._models_fetched_from_api:
            dynamic_models = self.fetch_models_from_api()
            if dynamic_models:
                # Update the available models with dynamic data
                self.available_ai_models.update(dynamic_models)
                logging.info(f"Updated model list with {len(dynamic_models)} models from API")
            # Mark as fetched regardless of success/failure to avoid repeated attempts
            self._models_fetched_from_api = True
        
        # Always return a copy of available models (either updated or hardcoded)
        return self.available_ai_models.copy()

    def fetch_models_from_api(self) -> Optional[Dict[str, Dict]]:
        """Fetch available models dynamically from Google Gemini API.
        
        Returns:
            Optional[Dict[str, Dict]]: Dictionary of models or None if failed.
        """
        try:
            if not self.google_api_key:
                logging.warning("No Google API key available for dynamic model fetching")
                return None

            from google import genai
            
            # Initialize client
            client = genai.Client(api_key=self.google_api_key)
            
            # List available models
            models_response = client.models.list()
            
            dynamic_models = {}
            
            # Process each model from the API
            for model in models_response:
                model_name = model.name
                
                # Filter for Gemini models only
                if 'gemini' in model_name.lower():
                    # Extract the model ID (e.g., "models/gemini-1.5-pro" -> "gemini-1.5-pro")
                    model_id = model_name.split('/')[-1] if '/' in model_name else model_name
                    
                    # Get model details
                    display_name = getattr(model, 'display_name', model_id.replace('-', ' ').title())
                    description = getattr(model, 'description', f"Google {display_name} model")
                    
                    # Determine rate limits based on model type (these are still approximate)
                    rate_limits = self._estimate_rate_limits_from_model_name(model_id)
                    
                    dynamic_models[model_id] = {
                        "name": display_name,
                        "description": description,
                        "rate_limit_requests": rate_limits["requests"],
                        "rate_limit_tokens": rate_limits["tokens"], 
                        "max_tokens": rate_limits["max_tokens"],
                        "tier": "free",  # Assume free tier for now
                        "api_name": model_name,  # Store full API name
                        "source": "api"  # Mark as dynamically fetched
                    }
            
            if dynamic_models:
                logging.info(f"Successfully fetched {len(dynamic_models)} models from API")
                return dynamic_models
            else:
                logging.warning("No Gemini models found in API response")
                return None
                
        except ImportError as e:
            logging.warning(f"Google API libraries not available: {e}")
            return None
        except Exception as e:
            logging.error(f"Error fetching models from API: {e}")
            return None

    def _estimate_rate_limits_from_model_name(self, model_id: str) -> Dict[str, int]:
        """Estimate rate limits based on model name patterns.
        
        Args:
            model_id (str): Model identifier.
            
        Returns:
            Dict[str, int]: Estimated rate limits.
        """
        # These are still estimates, but based on Google's published limits
        if 'pro' in model_id.lower():
            return {
                "requests": 2,      # Pro models have lower request limits
                "tokens": 32000,
                "max_tokens": 4096
            }
        elif 'flash' in model_id.lower():
            if 'lite' in model_id.lower():
                return {
                    "requests": 15,     # Flash Lite models
                    "tokens": 32000,
                    "max_tokens": 1000
                }
            else:
                return {
                    "requests": 15,     # Regular Flash models
                    "tokens": 1000000,
                    "max_tokens": 2048
                }
        else:
            # Default fallback
            return {
                "requests": 12,
                "tokens": 32000,
                "max_tokens": 1000
            }

    def update_model_settings(self, model_name: str) -> bool:
        """Update configuration based on selected AI model.
        
        Args:
            model_name (str): Name of the AI model to use.
            
        Returns:
            bool: True if model was updated successfully, False if model not found.
        """
        try:
            # Get available models (includes dynamic fetch)
            available_models = self.get_available_models()
            
            if model_name not in available_models:
                logging.warning(f"Unknown AI model: {model_name}")
                return False
        except Exception as e:
            logging.warning(f"Error getting available models: {e}")
            # Use hardcoded fallback models
            available_models = self.available_ai_models
            
            if model_name not in available_models:
                logging.warning(f"Unknown AI model: {model_name}")
                return False
        
        model_config = available_models[model_name]
        
        # Update model name
        self.ai_model_name = model_name
        
        # Update rate limiting based on model specifications
        self.gemini_rate_limit_requests = model_config["rate_limit_requests"]
        self.gemini_rate_limit_tokens = model_config["rate_limit_tokens"]
        self.ai_max_tokens = model_config["max_tokens"]
        
        logging.info(f"Updated configuration for model: {model_config['name']}")
        logging.info(f"Rate limits: {self.gemini_rate_limit_requests} req/min, {self.gemini_rate_limit_tokens} tokens/min")
        logging.info(f"Model source: {model_config.get('source', 'hardcoded')}")
        
        return True

    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently selected model.
        
        Returns:
            Dict[str, Any]: Current model information.
        """
        # Get available models (includes dynamic fetch)
        available_models = self.get_available_models()
        
        if self.ai_model_name in available_models:
            model_info = available_models[self.ai_model_name].copy()
            model_info["current_model"] = self.ai_model_name
            model_info["current_rate_limit_requests"] = self.gemini_rate_limit_requests
            model_info["current_rate_limit_tokens"] = self.gemini_rate_limit_tokens
            model_info["current_max_tokens"] = self.ai_max_tokens
            return model_info
        else:
            return {
                "current_model": self.ai_model_name,
                "name": "Unknown Model",
                "description": "Model configuration not found",
                "current_rate_limit_requests": self.gemini_rate_limit_requests,
                "current_rate_limit_tokens": self.gemini_rate_limit_tokens,
                "current_max_tokens": self.ai_max_tokens,
            }

    def detect_rate_limits_dynamically(self, model_name: str) -> Dict[str, int]:
        """Get rate limits for a specific model (simplified version).
        
        Simplified to return model-specific limits without complex dynamic detection.
        This provides the flexibility needed while avoiding over-engineering.
        
        Args:
            model_name (str): Name of the model to test.
            
        Returns:
            Dict[str, int]: Model rate limits.
        """
        # Simplified: Just return the configured limits for the model
        if model_name in self.available_ai_models:
            model_config = self.available_ai_models[model_name]
            return {
                "requests_per_minute": model_config["rate_limit_requests"],
                "tokens_per_minute": model_config["rate_limit_tokens"]
            }
        else:
            # Default fallback
            return {
                "requests_per_minute": 12,
                "tokens_per_minute": 32000
            }

    def refresh_available_models(self) -> bool:
        """Force refresh of available models from API.
        
        Returns:
            bool: True if refresh was successful, False otherwise.
        """
        try:
            # Reset the cache flag to force a new fetch
            self._models_fetched_from_api = False
            
            dynamic_models = self.fetch_models_from_api()
            if dynamic_models:
                # Update hardcoded models with dynamic data
                self.available_ai_models.update(dynamic_models)
                logging.info(f"Successfully refreshed models: {list(dynamic_models.keys())}")
                # Mark as fetched
                self._models_fetched_from_api = True
                return True
            else:
                logging.warning("Failed to refresh models from API")
                self._models_fetched_from_api = True  # Avoid repeated failures
                return False
        except Exception as e:
            logging.error(f"Error refreshing models: {e}")
            return False

    def get_cache_duration(self, data_type: str = "price") -> int:
        """Get appropriate cache duration for data type.

        Args:
            data_type (str): Data type ("price", "news", "ai", or app type "streamlit").

        Returns:
            int: Cache duration in seconds.
        """
        cache_mapping = {
            "price": self.default_cache_duration,
            "news": self.news_cache_duration,
            "ai": self.ai_cache_duration,
            "streamlit": self.default_cache_duration,
        }
        return cache_mapping.get(data_type.lower(), self.default_cache_duration)

    def validate_ai_parameters(self) -> bool:
        """Validate AI analysis parameter configuration.

        Returns:
            bool: True if configuration is valid.
        """
        return (
            self.news_articles_count > 0
            and self.max_news_age_hours > 0
            and 0.0 <= self.ai_temperature <= 2.0
            and self.ai_max_tokens > 0
            and 0.0 <= self.min_confidence_threshold <= 1.0
            and 0.0 <= self.high_confidence_threshold <= 1.0
            and self.min_confidence_threshold <= self.high_confidence_threshold
        )

    def validate_rate_limits(self) -> bool:
        """Validate rate limiting configuration.

        Returns:
            bool: True if rate limiting configuration is valid.
        """
        return (
            self.gemini_rate_limit_requests > 0
            and self.gemini_rate_limit_tokens > 0
            and self.newsapi_rate_limit_requests > 0
            and self.max_retries >= 0
            and self.retry_base_delay > 0
            and self.retry_max_delay >= self.retry_base_delay
            and self.api_timeout_seconds > 0
            and self.connection_timeout_seconds > 0
        )

    def get_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for retry attempts.

        Args:
            attempt (int): Current retry attempt number (1-based).

        Returns:
            float: Delay in seconds for the retry attempt.
        """
        delay = self.retry_base_delay * (2 ** (attempt - 1))
        return min(delay, self.retry_max_delay)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dict containing all configuration parameters (excluding sensitive API keys).
        """
        return {
            # AI Analysis Parameters
            "news_articles_count": self.news_articles_count,
            "max_news_age_hours": self.max_news_age_hours,
            "ai_model_name": self.ai_model_name,
            "ai_temperature": self.ai_temperature,
            "ai_max_tokens": self.ai_max_tokens,
            # Rate Limiting Configuration
            "gemini_rate_limit_requests": self.gemini_rate_limit_requests,
            "gemini_rate_limit_tokens": self.gemini_rate_limit_tokens,
            "newsapi_rate_limit_requests": self.newsapi_rate_limit_requests,
            "max_retries": self.max_retries,
            "retry_base_delay": self.retry_base_delay,
            "retry_max_delay": self.retry_max_delay,
            "api_timeout_seconds": self.api_timeout_seconds,
            "connection_timeout_seconds": self.connection_timeout_seconds,
            # Data Management
            "default_cache_duration": self.default_cache_duration,
            "news_cache_duration": self.news_cache_duration,
            "ai_cache_duration": self.ai_cache_duration,
            # Application Settings
            "default_ticker": self.default_ticker,
            "default_period_days": self.default_period_days,
            "streamlit_port": self.streamlit_port,
            "min_data_points": self.min_data_points,
            "max_date_range_years": self.max_date_range_years,
            "max_concurrent_updates": self.max_concurrent_updates,
            "chart_height": self.chart_height,
            "chart_theme": self.chart_theme,
            # AI Confidence Thresholds
            "min_confidence_threshold": self.min_confidence_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
        }


# Global configuration instance
CONFIG = AITradingConfig()


def get_config() -> AITradingConfig:
    """Get the global configuration instance.

    Returns:
        AITradingConfig: The global configuration object.
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

    if not CONFIG.validate_ai_parameters():
        raise ValueError("Invalid AI analysis parameter configuration")

    if not CONFIG.validate_rate_limits():
        raise ValueError("Invalid rate limiting configuration")


def reset_config() -> None:
    """Reset configuration to default values."""
    global CONFIG
    CONFIG = AITradingConfig()


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




# API Key Management Functions
def validate_environment() -> Dict[str, Any]:
    """Validate environment configuration and API key availability.

    Returns:
        Dict containing validation results and recommendations.
    """
    config = get_config()
    api_keys = config.validate_api_keys()

    validation_result = {
        "api_keys_valid": all(api_keys.values()),
        "api_key_status": api_keys,
        "configuration_valid": (
            config.validate_ai_parameters() and config.validate_rate_limits()
        ),
        "warnings": [],
        "errors": [],
    }

    # Check for missing API keys
    if not api_keys["google_api_key"]:
        validation_result["errors"].append(
            "GOOGLE_API_KEY environment variable not set. "
            "Required for Gemini AI analysis."
        )

    if not api_keys["newsapi_api_key"]:
        validation_result["errors"].append(
            "NEWSAPI_API_KEY environment variable not set. "
            "Required for news data fetching."
        )

    return validation_result


# Environment-specific overrides
def load_config_from_env() -> None:
    """Load configuration overrides from environment variables.

    Looks for environment variables with CONFIG_ prefix and updates
    the global configuration accordingly.
    """
    env_mappings = {
        "CONFIG_NEWS_ARTICLES_COUNT": ("news_articles_count", int),
        "CONFIG_AI_TEMPERATURE": ("ai_temperature", float),
        "CONFIG_MAX_RETRIES": ("max_retries", int),
        "CONFIG_CACHE_DURATION": ("default_cache_duration", int),
        "CONFIG_DEFAULT_TICKER": ("default_ticker", str),
        "CONFIG_STREAMLIT_PORT": ("streamlit_port", int),
    }

    updates = {}
    for env_var, (config_key, type_func) in env_mappings.items():
        if env_var in os.environ:
            try:
                updates[config_key] = type_func(os.environ[env_var])
            except ValueError:
                logging.warning(f"Invalid value for {env_var}, using default")

    if updates:
        update_config(**updates)
