"""
API Utilities for AI Trading Demo

This module provides centralized API validation patterns, rate limiting utilities,
and common API interaction patterns shared across all API integrations.

Author: AI Trading Demo Team
Version: 1.0 (Refactored for Code Deduplication)
"""

import datetime
import time
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import logging

from .errors import RetryConfig, handle_api_error, setup_logger


class APIProvider(Enum):
    """Supported API providers."""
    GOOGLE_GEMINI = "google_gemini"
    NEWS_API = "news_api"
    YAHOO_FINANCE = "yahoo_finance"


@dataclass
class APIKeyValidation:
    """API key validation result."""
    
    is_valid: bool
    provider: APIProvider
    key_format_valid: bool
    key_length_valid: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "is_valid": self.is_valid,
            "provider": self.provider.value,
            "key_format_valid": self.key_format_valid,
            "key_length_valid": self.key_length_valid,
            "error_message": self.error_message
        }


@dataclass 
class RateLimitStatus:
    """Rate limit status information."""
    
    requests_remaining: int
    tokens_remaining: Optional[int]
    reset_time: Optional[datetime.datetime]
    retry_after: Optional[float]
    is_exhausted: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "requests_remaining": self.requests_remaining,
            "tokens_remaining": self.tokens_remaining,
            "reset_time": self.reset_time.isoformat() if self.reset_time else None,
            "retry_after": self.retry_after,
            "is_exhausted": self.is_exhausted
        }


class APIValidator:
    """Centralized API validation utilities."""
    
    # API key validation patterns
    VALIDATION_PATTERNS = {
        APIProvider.GOOGLE_GEMINI: {
            "pattern": r"^[A-Za-z0-9_-]{39}$",
            "min_length": 39,
            "max_length": 39,
            "description": "Google API keys are typically 39 characters"
        },
        APIProvider.NEWS_API: {
            "pattern": r"^[a-f0-9]{32}$",
            "min_length": 32,
            "max_length": 32,
            "description": "NewsAPI keys are 32-character hexadecimal strings"
        }
    }
    
    # Common placeholder values that indicate invalid keys
    PLACEHOLDER_VALUES = {
        "your_google_gemini_api_key_here",
        "your_newsapi_key_here", 
        "XXXXXXXXXXXXXXXXXXXXXXX",
        "YOUR_API_KEY_HERE",
        "placeholder",
        "example",
        "test",
        ""
    }
    
    @classmethod
    def validate_api_key(
        cls, 
        api_key: Optional[str], 
        provider: APIProvider
    ) -> APIKeyValidation:
        """Validate API key format and structure.
        
        Args:
            api_key (Optional[str]): API key to validate.
            provider (APIProvider): API provider type.
            
        Returns:
            APIKeyValidation: Validation result.
        """
        if not api_key or api_key.strip() in cls.PLACEHOLDER_VALUES:
            return APIKeyValidation(
                is_valid=False,
                provider=provider,
                key_format_valid=False,
                key_length_valid=False,
                error_message="API key is missing or contains placeholder value"
            )
        
        api_key = api_key.strip()
        
        # Get validation rules for provider
        if provider not in cls.VALIDATION_PATTERNS:
            # Generic validation for unknown providers
            key_length_valid = len(api_key) >= 10
            key_format_valid = bool(re.match(r"^[A-Za-z0-9_-]+$", api_key))
        else:
            rules = cls.VALIDATION_PATTERNS[provider]
            key_length_valid = rules["min_length"] <= len(api_key) <= rules["max_length"]
            key_format_valid = bool(re.match(rules["pattern"], api_key))
        
        is_valid = key_length_valid and key_format_valid
        error_message = None
        
        if not is_valid:
            if not key_length_valid:
                error_message = f"API key length is invalid for {provider.value}"
            elif not key_format_valid:
                error_message = f"API key format is invalid for {provider.value}"
        
        return APIKeyValidation(
            is_valid=is_valid,
            provider=provider,
            key_format_valid=key_format_valid,
            key_length_valid=key_length_valid,
            error_message=error_message
        )
    
    @classmethod
    def validate_ticker_symbol(cls, ticker: str) -> Tuple[bool, Optional[str]]:
        """Validate stock ticker symbol format.
        
        Args:
            ticker (str): Ticker symbol to validate.
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if not ticker or not ticker.strip():
            return False, "Ticker symbol cannot be empty"
        
        ticker = ticker.strip().upper()
        
        # Basic ticker validation (1-5 characters, letters only for most cases)
        if not re.match(r"^[A-Z]{1,5}$", ticker):
            # Allow some special cases (e.g., BRK.A, BRK.B)
            if not re.match(r"^[A-Z]{1,4}\.[A-Z]$", ticker):
                return False, "Ticker must be 1-5 letters (e.g., AAPL, TSLA)"
        
        # Check for obviously invalid tickers
        invalid_tickers = {"TEST", "FAKE", "INVALID", "NULL", "NONE"}
        if ticker in invalid_tickers:
            return False, f"'{ticker}' is not a valid ticker symbol"
        
        return True, None
    
    @classmethod
    def validate_date_range(
        cls, 
        start_date: datetime.date, 
        end_date: datetime.date,
        max_range_days: int = 7300,  # ~20 years
        min_range_days: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """Validate date range for data requests.
        
        Args:
            start_date (datetime.date): Start date.
            end_date (datetime.date): End date.
            max_range_days (int): Maximum allowed range in days.
            min_range_days (int): Minimum required range in days.
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        today = datetime.date.today()
        
        # Check basic date logic
        if start_date >= end_date:
            return False, "Start date must be before end date"
        
        # Check if dates are in the future
        if start_date > today:
            return False, "Start date cannot be in the future"
        
        if end_date > today:
            return False, "End date cannot be in the future"
        
        # Check date range
        range_days = (end_date - start_date).days
        
        if range_days < min_range_days:
            return False, f"Date range too short (minimum {min_range_days} days)"
        
        if range_days > max_range_days:
            return False, f"Date range too long (maximum {max_range_days} days)"
        
        # Check if start date is too far in the past
        max_history_days = 7300  # ~20 years
        if (today - start_date).days > max_history_days:
            return False, f"Start date too far in the past (maximum {max_history_days} days)"
        
        return True, None
    
    @classmethod
    def validate_news_api_date_range(
        cls,
        start_date: datetime.date,
        end_date: datetime.date
    ) -> Tuple[bool, Optional[str]]:
        """Validate date range for NewsAPI (30-day limit for free tier).
        
        Args:
            start_date (datetime.date): Start date.
            end_date (datetime.date): End date.
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # First validate basic date range
        is_valid, error_msg = cls.validate_date_range(start_date, end_date)
        if not is_valid:
            return is_valid, error_msg
        
        # Check NewsAPI specific limitations (30 days for free tier)
        today = datetime.date.today()
        newsapi_limit_date = today - datetime.timedelta(days=30)
        
        if start_date < newsapi_limit_date:
            return False, f"NewsAPI free tier only supports last 30 days (earliest: {newsapi_limit_date})"
        
        return True, None


class RateLimiter:
    """Generic rate limiter with token bucket algorithm."""
    
    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: Optional[int] = None,
        burst_capacity: Optional[int] = None
    ):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute (int): Maximum requests per minute.
            tokens_per_minute (Optional[int]): Maximum tokens per minute.
            burst_capacity (Optional[int]): Burst capacity for requests.
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.burst_capacity = burst_capacity or min(10, requests_per_minute)
        
        # Token buckets
        self.request_tokens = float(self.burst_capacity)
        self.token_budget = float(tokens_per_minute) if tokens_per_minute else None
        
        # Timing
        self.last_update = time.time()
        self.lock = threading.Lock()
        
        # Rate calculation
        self.request_refill_rate = requests_per_minute / 60.0  # per second
        self.token_refill_rate = (tokens_per_minute / 60.0) if tokens_per_minute else None
    
    def can_make_request(self, tokens_needed: int = 0) -> Tuple[bool, Optional[float]]:
        """Check if request can be made within rate limits.
        
        Args:
            tokens_needed (int): Number of tokens needed for request.
            
        Returns:
            Tuple[bool, Optional[float]]: (can_proceed, wait_time_seconds)
        """
        with self.lock:
            self._refill_tokens()
            
            # Check request rate limit
            if self.request_tokens < 1:
                wait_time = (1 - self.request_tokens) / self.request_refill_rate
                return False, wait_time
            
            # Check token rate limit if applicable
            if self.token_budget is not None and tokens_needed > 0:
                if self.token_budget < tokens_needed:
                    wait_time = (tokens_needed - self.token_budget) / self.token_refill_rate
                    return False, wait_time
            
            return True, None
    
    def consume(self, tokens_used: int = 0) -> bool:
        """Consume rate limit resources.
        
        Args:
            tokens_used (int): Number of tokens consumed.
            
        Returns:
            bool: True if consumption successful.
        """
        with self.lock:
            self._refill_tokens()
            
            # Check if we can consume
            can_proceed, _ = self.can_make_request(tokens_used)
            if not can_proceed:
                return False
            
            # Consume resources
            self.request_tokens -= 1
            if self.token_budget is not None and tokens_used > 0:
                self.token_budget -= tokens_used
            
            return True
    
    def _refill_tokens(self) -> None:
        """Refill token buckets based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        
        # Refill request tokens
        self.request_tokens = min(
            self.burst_capacity,
            self.request_tokens + elapsed * self.request_refill_rate
        )
        
        # Refill token budget
        if self.token_budget is not None and self.token_refill_rate:
            max_tokens = self.tokens_per_minute
            self.token_budget = min(
                max_tokens,
                self.token_budget + elapsed * self.token_refill_rate
            )
        
        self.last_update = now
    
    def get_status(self) -> RateLimitStatus:
        """Get current rate limit status.
        
        Returns:
            RateLimitStatus: Current status information.
        """
        with self.lock:
            self._refill_tokens()
            
            # Calculate reset time (when bucket will be full)
            request_reset_time = None
            if self.request_tokens < self.burst_capacity:
                tokens_needed = self.burst_capacity - self.request_tokens
                seconds_to_reset = tokens_needed / self.request_refill_rate
                request_reset_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds_to_reset)
            
            return RateLimitStatus(
                requests_remaining=int(self.request_tokens),
                tokens_remaining=int(self.token_budget) if self.token_budget else None,
                reset_time=request_reset_time,
                retry_after=max(0, (1 - self.request_tokens) / self.request_refill_rate),
                is_exhausted=self.request_tokens < 1
            )


class APIConnectionTester:
    """Test API connections with timeout and retry logic."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize connection tester.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance.
        """
        self.logger = logger or setup_logger(__name__)
    
    def test_google_gemini_connection(self, api_key: str) -> Dict[str, Any]:
        """Test Google Gemini API connection.
        
        Args:
            api_key (str): Google API key.
            
        Returns:
            Dict[str, Any]: Connection test results.
        """
        result = {
            "connected": False,
            "api_key_valid": False,
            "error": None,
            "response_time": None,
            "model_available": False
        }
        
        try:
            # Validate key format first
            validation = APIValidator.validate_api_key(api_key, APIProvider.GOOGLE_GEMINI)
            result["api_key_valid"] = validation.is_valid
            
            if not validation.is_valid:
                result["error"] = validation.error_message
                return result
            
            # Test actual connection
            start_time = time.time()
            
            from google import genai
            client = genai.Client(api_key=api_key)
            
            # Try to list models as a connection test
            models_response = client.models.list()
            models = list(models_response)
            
            result["connected"] = True
            result["model_available"] = len(models) > 0
            result["response_time"] = time.time() - start_time
            
            self.logger.info(f"Google Gemini API connection successful ({len(models)} models available)")
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Google Gemini API connection failed: {e}")
        
        return result
    
    def test_news_api_connection(self, api_key: str) -> Dict[str, Any]:
        """Test NewsAPI connection.
        
        Args:
            api_key (str): NewsAPI key.
            
        Returns:
            Dict[str, Any]: Connection test results.
        """
        result = {
            "connected": False,
            "api_key_valid": False,
            "error": None,
            "response_time": None,
            "sources_available": 0
        }
        
        try:
            # Validate key format first
            validation = APIValidator.validate_api_key(api_key, APIProvider.NEWS_API)
            result["api_key_valid"] = validation.is_valid
            
            if not validation.is_valid:
                result["error"] = validation.error_message
                return result
            
            # Test actual connection
            start_time = time.time()
            
            from newsapi import NewsApiClient
            client = NewsApiClient(api_key=api_key)
            
            # Try to get sources as a connection test
            sources_response = client.get_sources(language="en", country="us")
            
            if sources_response and sources_response.get("status") == "ok":
                result["connected"] = True
                result["sources_available"] = len(sources_response.get("sources", []))
                result["response_time"] = time.time() - start_time
                
                self.logger.info(f"NewsAPI connection successful ({result['sources_available']} sources)")
            else:
                result["error"] = "API returned error status"
                
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"NewsAPI connection failed: {e}")
        
        return result


def create_rate_limiter_for_provider(provider: APIProvider, config: Dict[str, Any]) -> RateLimiter:
    """Create rate limiter configured for specific API provider.
    
    Args:
        provider (APIProvider): API provider.
        config (Dict[str, Any]): Configuration parameters.
        
    Returns:
        RateLimiter: Configured rate limiter.
    """
    if provider == APIProvider.GOOGLE_GEMINI:
        return RateLimiter(
            requests_per_minute=config.get("requests_per_minute", 12),
            tokens_per_minute=config.get("tokens_per_minute", 32000),
            burst_capacity=config.get("burst_capacity", 5)
        )
    elif provider == APIProvider.NEWS_API:
        # NewsAPI has daily limits, so we convert to per-minute
        daily_requests = config.get("requests_per_day", 1000)
        requests_per_minute = daily_requests / (24 * 60)  # Spread over day
        return RateLimiter(
            requests_per_minute=int(requests_per_minute),
            burst_capacity=config.get("burst_capacity", 10)
        )
    else:
        # Generic rate limiter
        return RateLimiter(
            requests_per_minute=config.get("requests_per_minute", 60),
            burst_capacity=config.get("burst_capacity", 10)
        )


def extract_error_info_from_response(response: Any, provider: APIProvider) -> Dict[str, Any]:
    """Extract error information from API response.
    
    Args:
        response: API response object.
        provider (APIProvider): API provider type.
        
    Returns:
        Dict[str, Any]: Extracted error information.
    """
    error_info = {
        "has_error": False,
        "error_type": "unknown",
        "error_message": None,
        "retry_after": None,
        "quota_exhausted": False
    }
    
    try:
        if provider == APIProvider.NEWS_API:
            if isinstance(response, dict):
                if response.get("status") == "error":
                    error_info["has_error"] = True
                    error_info["error_message"] = response.get("message", "Unknown error")
                    
                    # Check for specific error types
                    message = error_info["error_message"].lower()
                    if "rate limit" in message:
                        error_info["error_type"] = "rate_limit"
                    elif "quota" in message or "upgrade" in message:
                        error_info["error_type"] = "quota_exceeded"
                        error_info["quota_exhausted"] = True
                    elif "api key" in message or "unauthorized" in message:
                        error_info["error_type"] = "authentication"
                    
        elif provider == APIProvider.GOOGLE_GEMINI:
            # Google API errors are typically exceptions, not response dicts
            # This would be handled in exception handling elsewhere
            pass
    
    except Exception:
        # If we can't parse the error, just return default
        pass
    
    return error_info


# Convenience functions for common validations
def validate_trading_parameters(
    ticker: str,
    start_date: datetime.date,
    end_date: datetime.date,
    use_news_api: bool = False
) -> Tuple[bool, List[str]]:
    """Validate common trading analysis parameters.
    
    Args:
        ticker (str): Stock ticker symbol.
        start_date (datetime.date): Analysis start date.
        end_date (datetime.date): Analysis end date.
        use_news_api (bool): Whether NewsAPI will be used.
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
    """
    errors = []
    
    # Validate ticker
    ticker_valid, ticker_error = APIValidator.validate_ticker_symbol(ticker)
    if not ticker_valid:
        errors.append(f"Ticker: {ticker_error}")
    
    # Validate date range
    if use_news_api:
        date_valid, date_error = APIValidator.validate_news_api_date_range(start_date, end_date)
    else:
        date_valid, date_error = APIValidator.validate_date_range(start_date, end_date)
    
    if not date_valid:
        errors.append(f"Date range: {date_error}")
    
    return len(errors) == 0, errors


def format_api_error_for_ui(error_info: Dict[str, Any], operation: str = "API request") -> str:
    """Format API error information for user interface display.
    
    Args:
        error_info (Dict[str, Any]): Error information from handle_api_error.
        operation (str): Description of the operation that failed.
        
    Returns:
        str: Formatted error message for UI.
    """
    if error_info.get("error_type") == "quota_exceeded":
        return f"API quota exhausted for {operation}. Please try again later or upgrade your plan."
    elif error_info.get("error_type") == "rate_limit":
        wait_time = error_info.get("wait_time", 60)
        return f"Rate limit reached for {operation}. Please wait {wait_time} seconds before retrying."
    elif error_info.get("error_type") == "auth_error":
        return f"Authentication failed for {operation}. Please check your API key configuration."
    elif error_info.get("error_type") == "network_error":
        return f"Network error during {operation}. Please check your internet connection and try again."
    elif error_info.get("error_type") == "validation_error":
        return f"Invalid parameters for {operation}. Please check your input values."
    else:
        return f"An error occurred during {operation}: {error_info.get('error_message', 'Unknown error')}"