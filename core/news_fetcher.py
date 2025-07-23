"""
News Data Fetcher for AI Trading Demo

This module handles news data fetching from NewsAPI with comprehensive rate limiting,
caching, error handling, and retry mechanisms for reliable news data acquisition.

Author: AI Trading Demo Team
Version: 2.0 (AI-Powered Hybrid Architecture)
"""

import datetime
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import threading
from functools import lru_cache
import json

from newsapi import NewsApiClient
import requests

from .config import get_config
from utils.errors import (
    setup_logger, handle_api_error, RetryConfig,
    API_REQUEST_RETRY_CONFIG
)
from utils.api import APIValidator, APIProvider, RateLimiter


# Configure logging
logger = setup_logger(__name__)


@dataclass
class NewsArticle:
    """Data class representing a news article."""
    
    source: str
    title: str
    description: Optional[str]
    url: str
    published_at: datetime.datetime
    content: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary format.
        
        Returns:
            Dict containing article data.
        """
        return {
            "source": self.source,
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "content": self.content,
        }


class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, max_tokens: int, refill_rate: float):
        """Initialize token bucket.
        
        Args:
            max_tokens (int): Maximum number of tokens in bucket.
            refill_rate (float): Tokens per second refill rate.
        """
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.refill_rate = refill_rate
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from the bucket.
        
        Args:
            tokens (int): Number of tokens to consume.
            
        Returns:
            bool: True if tokens were consumed, False if insufficient tokens.
        """
        with self.lock:
            now = time.time()
            # Add tokens based on time elapsed
            time_passed = now - self.last_update
            self.tokens = min(
                self.max_tokens,
                self.tokens + time_passed * self.refill_rate
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_for_tokens(self, tokens: int = 1) -> float:
        """Calculate wait time needed for tokens to be available.
        
        Args:
            tokens (int): Number of tokens needed.
            
        Returns:
            float: Wait time in seconds.
        """
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.refill_rate


class NewsCache:
    """Simple in-memory cache for news data."""
    
    def __init__(self, default_ttl: int = 1800):
        """Initialize news cache.
        
        Args:
            default_ttl (int): Default time-to-live in seconds.
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[List[NewsArticle]]:
        """Get cached news data.
        
        Args:
            key (str): Cache key.
            
        Returns:
            Optional[List[NewsArticle]]: Cached articles if available and valid.
        """
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() < entry["expires"]:
                    return entry["data"]
                else:
                    del self.cache[key]
            return None
    
    def set(self, key: str, data: List[NewsArticle], ttl: Optional[int] = None) -> None:
        """Set cached news data.
        
        Args:
            key (str): Cache key.
            data (List[NewsArticle]): Articles to cache.
            ttl (Optional[int]): Time-to-live in seconds.
        """
        with self.lock:
            expires = time.time() + (ttl or self.default_ttl)
            self.cache[key] = {
                "data": data,
                "expires": expires,
                "cached_at": time.time()
            }
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self.lock:
            self.cache.clear()
    
    def cleanup(self) -> None:
        """Remove expired cache entries."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time >= entry["expires"]
            ]
            for key in expired_keys:
                del self.cache[key]


class NewsFetcher:
    """News data fetcher with rate limiting and caching."""
    
    def __init__(self, config=None):
        """Initialize news fetcher.
        
        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.client: Optional[NewsApiClient] = None
        self.cache = NewsCache(default_ttl=self.config.news_cache_duration)
        
        # Initialize rate limiters (requests per day = requests per 24*60*60 seconds)
        requests_per_second = self.config.newsapi_rate_limit_requests / (24 * 60 * 60)
        self.rate_limiter = TokenBucket(
            max_tokens=min(10, self.config.newsapi_rate_limit_requests),  # Burst capacity
            refill_rate=requests_per_second
        )
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize NewsAPI client with proper error handling."""
        try:
            if not self.config.newsapi_api_key:
                logger.error("NewsAPI API key not available")
                return
            
            self.client = NewsApiClient(api_key=self.config.newsapi_api_key)
            logger.info("NewsAPI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NewsAPI client: {e}")
            self.client = None
    
    def _make_request_with_retry(self, func, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Make API request with retry logic and rate limiting.
        
        Args:
            func: Function to call.
            *args: Function arguments.
            **kwargs: Function keyword arguments.
            
        Returns:
            Optional[Dict[str, Any]]: API response or None if failed.
        """
        if not self.client:
            logger.error("NewsAPI client not initialized")
            return None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Check rate limit
                if not self.rate_limiter.consume():
                    wait_time = self.rate_limiter.wait_for_tokens()
                    if wait_time > 60:  # If wait time is too long, fail
                        logger.warning(f"Rate limit exceeded. Would need to wait {wait_time:.1f} seconds")
                        return None
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                    time.sleep(wait_time)
                
                # Make the request
                logger.debug(f"Making API request (attempt {attempt})")
                response = func(*args, **kwargs)
                
                if response and response.get("status") == "ok":
                    logger.debug("API request successful")
                    return response
                else:
                    error_msg = response.get("message", "Unknown error") if response else "No response"
                    logger.warning(f"API request failed: {error_msg}")
                    
                    # Check if this is a quota/rate limit error
                    if response and "rate limit" in error_msg.lower():
                        logger.error("API rate limit exceeded")
                        return None
                    
                    if response and "quota" in error_msg.lower():
                        logger.error("API quota exhausted")
                        return None
                
            except requests.exceptions.RequestException as e:
                error_info = handle_api_error(e, "NewsAPI request", logger, attempt, self.config.max_retries)
                if error_info["retry_recommended"] and attempt < self.config.max_retries:
                    time.sleep(error_info["wait_time"])
                else:
                    logger.error("Max retries reached for network error")
                    return None
                
            except Exception as e:
                handle_api_error(e, "NewsAPI request", logger, attempt, self.config.max_retries)
                return None
        
        return None
    
    def _parse_articles(self, api_response: Dict[str, Any]) -> List[NewsArticle]:
        """Parse API response into NewsArticle objects.
        
        Args:
            api_response (Dict[str, Any]): Raw API response.
            
        Returns:
            List[NewsArticle]: Parsed articles.
        """
        articles = []
        raw_articles = api_response.get("articles", [])
        
        for article_data in raw_articles:
            try:
                # Parse published date
                published_str = article_data.get("publishedAt")
                if published_str:
                    # Remove 'Z' and parse ISO format
                    published_str = published_str.replace("Z", "+00:00")
                    published_at = datetime.datetime.fromisoformat(published_str)
                else:
                    published_at = datetime.datetime.now(datetime.timezone.utc)
                
                article = NewsArticle(
                    source=article_data.get("source", {}).get("name", "Unknown"),
                    title=article_data.get("title", ""),
                    description=article_data.get("description"),
                    url=article_data.get("url", ""),
                    published_at=published_at,
                    content=article_data.get("content")
                )
                articles.append(article)
                
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(articles)} articles")
        return articles
    
    def _generate_cache_key(self, ticker: str = "", **kwargs) -> str:
        """Generate cache key for news query.
        
        Args:
            ticker (str): Stock ticker (optional for general news).
            **kwargs: Additional query parameters.
            
        Returns:
            str: Cache key.
        """
        # Create a stable cache key from parameters
        key_parts = [ticker] if ticker else ["general"]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)
    
    def get_stock_news(
        self,
        ticker: str,
        max_age_hours: Optional[int] = None,
        max_articles: Optional[int] = None,
        use_cache: bool = True,
        specific_date: Optional[datetime.date] = None
    ) -> List[NewsArticle]:
        """Fetch news articles related to a stock ticker.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., "AAPL").
            max_age_hours (Optional[int]): Maximum age of articles in hours.
            max_articles (Optional[int]): Maximum number of articles to return.
            use_cache (bool): Whether to use cached results.
            specific_date (Optional[datetime.date]): Specific date to fetch news for.
            
        Returns:
            List[NewsArticle]: List of news articles.
        """
        # Use config defaults if not specified
        max_age_hours = max_age_hours or self.config.max_news_age_hours
        max_articles = max_articles or self.config.news_articles_count
        
        # Check cache first
        cache_key = self._generate_cache_key(
            ticker=ticker,
            max_age_hours=max_age_hours,
            max_articles=max_articles,
            specific_date=specific_date.isoformat() if specific_date else None
        )
        
        if use_cache:
            cached_articles = self.cache.get(cache_key)
            if cached_articles:
                logger.debug(f"Returning {len(cached_articles)} cached articles for {ticker}")
                return cached_articles[:max_articles]
        
        logger.info(f"Fetching news for ticker: {ticker}" + (f" on {specific_date}" if specific_date else ""))
        
        # Calculate date range
        if specific_date:
            # Use specific date with a tight window
            start_date = datetime.datetime.combine(specific_date, datetime.time.min, datetime.timezone.utc)
            end_date = start_date + datetime.timedelta(days=1)
        else:
            # Use time-based range
            end_date = datetime.datetime.now(datetime.timezone.utc)
            start_date = end_date - datetime.timedelta(hours=max_age_hours)
        
        # Create broader query to find relevant news
        query = f"{ticker} OR {ticker.lower()}"
        
        try:
            # Make API request - remove sort_by to get more diverse dates
            response = self._make_request_with_retry(
                self.client.get_everything,
                q=query,
                from_param=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                language="en",
                # Removed sort_by to get articles from different dates, not just latest
                page_size=min(max_articles * 2, 100)  # Fetch extra to filter
            )
            
            if not response:
                logger.warning(f"No response from NewsAPI for ticker {ticker}")
                # Fallback to general business news
                logger.info("Falling back to general business news")
                return self.get_general_market_news("business", max_articles=max_articles, use_cache=use_cache)
            
            # Parse articles
            articles = self._parse_articles(response)
            
            # For debugging: return all articles, let AI do the filtering
            filtered_articles = articles  # self._filter_relevant_articles(articles, ticker)
            
            # Limit results
            result_articles = filtered_articles[:max_articles]
            
            # If no articles found, fallback to general business news
            if not result_articles:
                logger.info(f"No specific articles found for {ticker}, falling back to general business news")
                return self.get_general_market_news("business", max_articles=max_articles, use_cache=use_cache)
            
            # Cache results
            if use_cache and result_articles:
                self.cache.set(cache_key, result_articles)
            
            logger.info(f"Fetched {len(result_articles)} articles for {ticker}")
            return result_articles
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []
    
    def _filter_relevant_articles(self, articles: List[NewsArticle], ticker: str) -> List[NewsArticle]:
        """Filter articles to find those most relevant to the ticker.
        
        Args:
            articles (List[NewsArticle]): All articles.
            ticker (str): Stock ticker symbol.
            
        Returns:
            List[NewsArticle]: Filtered articles.
        """
        if not articles:
            return []
        
        ticker_lower = ticker.lower()
        relevant_articles = []
        
        for article in articles:
            # Check if ticker appears in title or description
            title = (article.title or "").lower()
            description = (article.description or "").lower()
            
            if (ticker_lower in title or 
                ticker_lower in description or
                # Add company name mapping if needed
                self._is_likely_relevant(title + " " + description, ticker)):
                relevant_articles.append(article)
        
        # If we have few relevant articles, include more general ones
        if len(relevant_articles) < 3:
            relevant_articles = articles[:10]  # Take first 10 as fallback
        
        return relevant_articles
    
    def _is_likely_relevant(self, text: str, ticker: str) -> bool:
        """Check if text is likely relevant to ticker using simple heuristics.
        
        Args:
            text (str): Text to check.
            ticker (str): Stock ticker.
            
        Returns:
            bool: True if likely relevant.
        """
        # Simple relevance check - could be enhanced with ML
        business_keywords = [
            "stock", "shares", "trading", "market", "earnings", 
            "revenue", "profit", "loss", "financial", "business"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in business_keywords)
    
    def get_general_market_news(
        self,
        category: str = "business",
        max_articles: Optional[int] = None,
        use_cache: bool = True
    ) -> List[NewsArticle]:
        """Fetch general market/business news.
        
        Args:
            category (str): News category (business, technology, etc.).
            max_articles (Optional[int]): Maximum number of articles.
            use_cache (bool): Whether to use cached results.
            
        Returns:
            List[NewsArticle]: List of news articles.
        """
        max_articles = max_articles or self.config.news_articles_count
        
        cache_key = self._generate_cache_key(
            ticker="",
            category=category,
            max_articles=max_articles,
            type="general"
        )
        
        if use_cache:
            cached_articles = self.cache.get(cache_key)
            if cached_articles:
                logger.debug(f"Returning {len(cached_articles)} cached general articles")
                return cached_articles[:max_articles]
        
        logger.info(f"Fetching general {category} news")
        
        try:
            response = self._make_request_with_retry(
                self.client.get_top_headlines,
                category=category,
                language="en",
                page_size=max_articles
            )
            
            if not response:
                logger.warning(f"No response from NewsAPI for category {category}")
                return []
            
            articles = self._parse_articles(response)
            
            if use_cache and articles:
                self.cache.set(cache_key, articles)
            
            logger.info(f"Fetched {len(articles)} general articles")
            return articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Error fetching general news: {e}")
            return []
    
    def validate_connection(self) -> Dict[str, Any]:
        """Validate NewsAPI connection and return status.
        
        Returns:
            Dict[str, Any]: Connection status and information.
        """
        # Use centralized API key validation
        api_validation = APIValidator.validate_api_key(
            self.config.newsapi_api_key,
            APIProvider.NEWS_API
        )
        
        status = {
            "connected": False,
            "api_key_valid": api_validation.is_valid,
            "client_initialized": self.client is not None,
            "rate_limit_tokens": self.rate_limiter.tokens,
            "cache_entries": len(self.cache.cache),
            "error": api_validation.error_message if not api_validation.is_valid else None
        }
        
        if not self.client:
            status["error"] = "Client not initialized"
            return status
        
        try:
            # Make a minimal test request
            response = self._make_request_with_retry(
                self.client.get_sources,
                language="en",
                country="us"
            )
            
            if response and response.get("status") == "ok":
                status["connected"] = True
                status["sources_available"] = len(response.get("sources", []))
            else:
                status["error"] = "API request failed"
                
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    def clear_cache(self) -> None:
        """Clear all cached news data."""
        self.cache.clear()
        logger.info("News cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics.
        """
        return {
            "entries": len(self.cache.cache),
            "default_ttl": self.cache.default_ttl,
            "rate_limit_tokens": self.rate_limiter.tokens,
            "max_rate_limit_tokens": self.rate_limiter.max_tokens
        }


# Global news fetcher instance
_news_fetcher: Optional[NewsFetcher] = None


def get_news_fetcher() -> NewsFetcher:
    """Get the global news fetcher instance.
    
    Returns:
        NewsFetcher: The global news fetcher instance.
    """
    global _news_fetcher
    if _news_fetcher is None:
        _news_fetcher = NewsFetcher()
    return _news_fetcher


def fetch_stock_news(ticker: str, max_articles: int = 10) -> List[NewsArticle]:
    """Convenience function to fetch stock news.
    
    Args:
        ticker (str): Stock ticker symbol.
        max_articles (int): Maximum number of articles.
        
    Returns:
        List[NewsArticle]: List of news articles.
    """
    fetcher = get_news_fetcher()
    return fetcher.get_stock_news(ticker, max_articles=max_articles)


def validate_news_api() -> Dict[str, Any]:
    """Convenience function to validate NewsAPI connection.
    
    Returns:
        Dict[str, Any]: Connection validation results.
    """
    fetcher = get_news_fetcher()
    return fetcher.validate_connection()