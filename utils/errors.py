"""
Error Handler Utilities for AI Trading Demo

This module provides centralized error handling patterns, logging utilities,
and retry mechanisms shared across all modules in the application.

Author: AI Trading Demo Team
Version: 1.0 (Refactored for Code Deduplication)
"""

import logging
import time
import functools
from typing import Any, Callable, Dict, Optional, Union, TypeVar, Type
from dataclasses import dataclass
import traceback

# Type hints for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class ErrorContext:
    """Context information for error handling."""
    
    operation: str
    module: str
    retry_attempt: int = 0
    max_retries: int = 3
    additional_info: Optional[Dict[str, Any]] = None


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        backoff_jitter: float = 0.1
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts (int): Maximum retry attempts.
            base_delay (float): Base delay in seconds.
            max_delay (float): Maximum delay in seconds.
            exponential_base (float): Base for exponential backoff.
            backoff_jitter (float): Jitter factor for randomization.
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.backoff_jitter = backoff_jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with exponential backoff.
        
        Args:
            attempt (int): Current retry attempt (1-based).
            
        Returns:
            float: Delay in seconds.
        """
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        
        # Add jitter to prevent thundering herd
        import random
        jitter = random.uniform(-self.backoff_jitter, self.backoff_jitter)
        delay *= (1 + jitter)
        
        return min(delay, self.max_delay)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting.
    
    Args:
        name (str): Logger name.
        level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: ErrorContext,
    include_traceback: bool = False
) -> None:
    """Log error with context information.
    
    Args:
        logger (logging.Logger): Logger instance.
        error (Exception): Exception to log.
        context (ErrorContext): Error context.
        include_traceback (bool): Whether to include traceback.
    """
    error_msg = (
        f"Error in {context.operation} ({context.module}): {str(error)}"
    )
    
    if context.retry_attempt > 0:
        error_msg += f" (Attempt {context.retry_attempt}/{context.max_retries})"
    
    if context.additional_info:
        error_msg += f" | Additional info: {context.additional_info}"
    
    logger.error(error_msg)
    
    if include_traceback:
        logger.error(f"Traceback: {traceback.format_exc()}")


def handle_api_error(
    error: Exception,
    operation: str,
    logger: logging.Logger,
    attempt: int = 0,
    max_attempts: int = 3
) -> Dict[str, Any]:
    """Handle API-specific errors with standardized patterns.
    
    Args:
        error (Exception): The API error.
        operation (str): Description of the operation.
        logger (logging.Logger): Logger instance.
        attempt (int): Current attempt number.
        max_attempts (int): Maximum attempts.
        
    Returns:
        Dict[str, Any]: Error information dictionary.
    """
    error_str = str(error).lower()
    error_info = {
        "success": False,
        "error_type": "unknown",
        "error_message": str(error),
        "retry_recommended": False,
        "wait_time": 0,
        "attempt": attempt,
        "max_attempts": max_attempts
    }
    
    # Categorize common API errors
    if "quota" in error_str or "rate limit" in error_str:
        error_info.update({
            "error_type": "quota_exceeded",
            "retry_recommended": True,
            "wait_time": min(300, 60 * (attempt + 1))  # Exponential backoff, max 5 minutes
        })
        logger.warning(f"{operation}: API quota exhausted (attempt {attempt})")
    
    elif "network" in error_str or "connection" in error_str or "timeout" in error_str:
        error_info.update({
            "error_type": "network_error",
            "retry_recommended": True,
            "wait_time": min(30, 5 * (attempt + 1))
        })
        logger.warning(f"{operation}: Network error (attempt {attempt})")
    
    elif "authentication" in error_str or "unauthorized" in error_str or "api key" in error_str:
        error_info.update({
            "error_type": "auth_error",
            "retry_recommended": False
        })
        logger.error(f"{operation}: Authentication error - check API keys")
    
    elif "not found" in error_str or "invalid" in error_str:
        error_info.update({
            "error_type": "validation_error",
            "retry_recommended": False
        })
        logger.error(f"{operation}: Validation error - check input parameters")
    
    else:
        error_info.update({
            "error_type": "generic_error",
            "retry_recommended": attempt < max_attempts - 1,
            "wait_time": min(10, 2 * (attempt + 1))
        })
        logger.error(f"{operation}: Unexpected error (attempt {attempt}): {error}")
    
    return error_info


def retry_with_backoff(
    retry_config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        retry_config (Optional[RetryConfig]): Retry configuration.
        exceptions (tuple): Exceptions to retry on.
        logger (Optional[logging.Logger]): Logger for retry messages.
        
    Returns:
        Callable: Decorated function.
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == retry_config.max_attempts:
                        if logger:
                            logger.error(f"Max retries ({retry_config.max_attempts}) reached for {func.__name__}")
                        raise e
                    
                    delay = retry_config.get_delay(attempt)
                    if logger:
                        logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt}/{retry_config.max_attempts})")
                    
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def safe_execute(
    func: Callable[[], T],
    default_value: T,
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    log_errors: bool = True
) -> T:
    """Safely execute a function with error handling.
    
    Args:
        func (Callable): Function to execute.
        default_value: Value to return on error.
        operation_name (str): Name of operation for logging.
        logger (Optional[logging.Logger]): Logger instance.
        log_errors (bool): Whether to log errors.
        
    Returns:
        T: Function result or default value.
    """
    try:
        return func()
    except Exception as e:
        if log_errors and logger:
            logger.error(f"Error in {operation_name}: {str(e)}")
        return default_value


def validate_and_handle_response(
    response: Any,
    expected_type: Type[T],
    operation: str,
    logger: logging.Logger,
    allow_none: bool = False
) -> Optional[T]:
    """Validate API response and handle common issues.
    
    Args:
        response: The response to validate.
        expected_type: Expected response type.
        operation (str): Operation description.
        logger (logging.Logger): Logger instance.
        allow_none (bool): Whether None is acceptable.
        
    Returns:
        Optional[T]: Validated response or None.
    """
    if response is None:
        if allow_none:
            return None
        logger.error(f"{operation}: Received None response")
        return None
    
    if not isinstance(response, expected_type):
        logger.error(f"{operation}: Expected {expected_type}, got {type(response)}")
        return None
    
    return response


def create_error_response(
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized error response dictionary.
    
    Args:
        error_type (str): Type of error.
        message (str): Error message.
        details (Optional[Dict]): Additional error details.
        
    Returns:
        Dict[str, Any]: Standardized error response.
    """
    response = {
        "success": False,
        "error": True,
        "error_type": error_type,
        "message": message,
        "timestamp": time.time()
    }
    
    if details:
        response["details"] = details
    
    return response


def create_success_response(
    data: Any,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized success response dictionary.
    
    Args:
        data: Response data.
        message (Optional[str]): Success message.
        metadata (Optional[Dict]): Additional metadata.
        
    Returns:
        Dict[str, Any]: Standardized success response.
    """
    response = {
        "success": True,
        "error": False,
        "data": data,
        "timestamp": time.time()
    }
    
    if message:
        response["message"] = message
    
    if metadata:
        response["metadata"] = metadata
    
    return response


class ErrorHandler:
    """Context manager for error handling with logging and retry logic."""
    
    def __init__(
        self,
        operation: str,
        logger: logging.Logger,
        retry_config: Optional[RetryConfig] = None,
        suppress_errors: bool = False
    ):
        """Initialize error handler.
        
        Args:
            operation (str): Operation description.
            logger (logging.Logger): Logger instance.
            retry_config (Optional[RetryConfig]): Retry configuration.
            suppress_errors (bool): Whether to suppress exceptions.
        """
        self.operation = operation
        self.logger = logger
        self.retry_config = retry_config or RetryConfig()
        self.suppress_errors = suppress_errors
        self.error_info: Optional[Dict[str, Any]] = None
        self.success = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_info = handle_api_error(
                exc_val,
                self.operation,
                self.logger
            )
            
            if self.suppress_errors:
                return True  # Suppress the exception
        else:
            self.success = True
        
        return False


# Pre-configured error handlers for common operations
DATA_FETCH_RETRY_CONFIG = RetryConfig(max_attempts=3, base_delay=2.0, max_delay=30.0)
API_REQUEST_RETRY_CONFIG = RetryConfig(max_attempts=5, base_delay=1.0, max_delay=60.0)
VALIDATION_RETRY_CONFIG = RetryConfig(max_attempts=2, base_delay=0.5, max_delay=5.0)


# Convenience functions for common error patterns
def handle_data_fetch_error(error: Exception, operation: str, logger: logging.Logger) -> Dict[str, Any]:
    """Handle data fetching errors with appropriate categorization."""
    return handle_api_error(error, f"Data fetch: {operation}", logger)


def handle_validation_error(error: Exception, field: str, logger: logging.Logger) -> Dict[str, Any]:
    """Handle input validation errors."""
    return handle_api_error(error, f"Validation: {field}", logger)


def handle_processing_error(error: Exception, operation: str, logger: logging.Logger) -> Dict[str, Any]:
    """Handle data processing errors."""
    return handle_api_error(error, f"Processing: {operation}", logger)