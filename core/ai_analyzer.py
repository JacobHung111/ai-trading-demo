"""
AI Analysis Module for AI Trading Demo

This module handles AI-powered sentiment analysis using Google Gemini API with
comprehensive prompt engineering, JSON response parsing, and error handling.

Author: AI Trading Demo Team
Version: 2.0 (AI-Powered Hybrid Architecture)
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import threading
from functools import lru_cache

from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

from .config import get_config
from .news_fetcher import NewsArticle
from utils.errors import (
    setup_logger, handle_api_error, RetryConfig,
    retry_with_backoff, API_REQUEST_RETRY_CONFIG
)
from utils.api import APIValidator, APIProvider


# Configure logging
logger = setup_logger(__name__)


@dataclass
class AIAnalysisResult:
    """Container for AI analysis results with utility methods."""
    
    signal: str  # "BUY", "SELL", or "HOLD"
    confidence: float  # 0.0 to 1.0
    rationale: str  # AI's reasoning
    model_used: str  # Model name used for analysis
    analysis_timestamp: float  # Unix timestamp
    raw_response: Optional[str] = None  # Raw AI response for debugging
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to dictionary format.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the analysis result.
        """
        return {
            "signal": self.signal,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "model_used": self.model_used,
            "analysis_timestamp": self.analysis_timestamp,
            "raw_response": self.raw_response
        }
    
    def get_numerical_signal(self) -> int:
        """Convert signal to numerical format for compatibility.
        
        Returns:
            int: 1 for BUY, -1 for SELL, 0 for HOLD.
        """
        signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
        return signal_map.get(self.signal, 0)


class AIAnalyzer:
    """AI-powered sentiment analysis using Google Gemini API."""
    
    def __init__(self, config=None):
        """Initialize AI analyzer.
        
        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.client: Optional[genai.Client] = None
        self.model_name = self.config.ai_model_name
        
        # Rate limiting - integrate with existing rate limiter if needed
        self.last_request_time = 0.0
        self.request_lock = threading.Lock()
        
        self._initialize_client()
    
    def update_model(self, model_name: str) -> bool:
        """Update the AI model being used.
        
        Args:
            model_name (str): Name of the new model to use.
            
        Returns:
            bool: True if model was updated successfully.
        """
        try:
            # Update configuration
            if self.config.update_model_settings(model_name):
                self.model_name = model_name
                logger.info(f"AI analyzer updated to use model: {model_name}")
                
                # Re-initialize client if needed (for different model endpoints)
                # For Gemini models, the client remains the same but model name changes
                return True
            else:
                logger.error(f"Failed to update to unknown model: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating AI model: {e}")
            return False
    
    def _initialize_client(self) -> None:
        """Initialize Gemini client with proper error handling."""
        try:
            if not self.config.google_api_key:
                logger.error("Google API key not available")
                return
            
            # Initialize client with API key
            self.client = genai.Client(api_key=self.config.google_api_key)
            logger.info(f"Gemini AI client initialized successfully with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None
    
    def _construct_analysis_prompt(
        self, 
        ticker: str,
        articles: List[NewsArticle], 
        context: Optional[str] = None
    ) -> str:
        """Construct analysis prompt for AI model.
        
        Args:
            ticker (str): Stock ticker symbol.
            articles (List[NewsArticle]): News articles to analyze.
            context (str, optional): Additional context for analysis.
            
        Returns:
            str: Formatted prompt for AI analysis.
        """
        # Base prompt with clear instructions and strict JSON format requirements
        prompt = f"""As a professional financial analyst, analyze the following news headlines for the stock ticker {ticker}. 
Based on the sentiment and potential impact on stock price, provide a trading recommendation.

CRITICAL: Your response MUST be a single, valid JSON object with NO trailing commas and exactly these keys:
- "signal": Either "BUY", "SELL", or "HOLD"
- "confidence": A number between 0.0 and 1.0 representing your confidence  
- "reasoning": A brief explanation (2-3 sentences) of your reasoning

JSON FORMAT EXAMPLE:
{{"signal": "HOLD", "confidence": 0.65, "reasoning": "Mixed signals in the news suggest a cautious approach."}}

Consider factors like:
- Earnings performance and financial metrics
- Market sentiment and investor reaction
- Company developments and strategic changes
- Industry trends and competitive position

Recent news headlines for {ticker}:

"""
        
        # Add news articles to prompt
        for i, article in enumerate(articles, 1):
            # Build source information
            source_info = f"Source: {article.source}"
            if article.published_at:
                source_info += f" ({article.published_at.strftime('%Y-%m-%d')})"
            
            prompt += f"{i}. {source_info}\n"
            prompt += f"   Title: {article.title}\n"
            
            if article.description:
                prompt += f"   Summary: {article.description}\n"
            
            prompt += "\n"
        
        # Add additional context if provided
        context_str = ""
        if context:
            context_str = f"\nAdditional Context: {context}\n"
        
        prompt += f"""{context_str}
Based on this news analysis, provide your trading recommendation as a JSON response:
"""
        
        return prompt
    
    def _make_ai_request(self, prompt: str) -> Optional[str]:
        """Make AI request with rate limiting and retry logic.
        
        Args:
            prompt (str): The prompt to send to AI model.
            
        Returns:
            Optional[str]: AI response text or None if failed.
        """
        if not self.client:
            logger.error("Gemini client not initialized")
            return None
        
        # Enhanced rate limiting for Gemini Free Tier
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            min_interval = 60.0 / self.config.gemini_rate_limit_requests  # 5 seconds minimum between requests
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds to avoid quota exhaustion")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug(f"Making AI request (attempt {attempt})")
                
                # Configure generation parameters
                generation_config = types.GenerateContentConfig(
                    temperature=self.config.ai_temperature,
                    max_output_tokens=self.config.ai_max_tokens,
                )
                
                # Make the request using the new API
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generation_config
                )
                
                # Extract response text
                if response and hasattr(response, 'text') and response.text:
                    logger.debug("AI request successful")
                    return response.text.strip()
                else:
                    logger.warning(f"Empty response from AI model (attempt {attempt})")
                    
            except google_exceptions.ResourceExhausted as e:
                error_info = handle_api_error(e, "Gemini AI request", logger, attempt, self.config.max_retries)
                if error_info["retry_recommended"] and attempt < self.config.max_retries:
                    time.sleep(error_info["wait_time"])
                else:
                    logger.error("Max retries exceeded due to quota exhaustion")
                    break
                    
            except google_exceptions.GoogleAPICallError as e:
                error_info = handle_api_error(e, "Gemini AI request", logger, attempt, self.config.max_retries)
                if error_info["retry_recommended"] and attempt < self.config.max_retries:
                    time.sleep(error_info["wait_time"])
                else:
                    logger.error("Max retries exceeded due to API errors")
                    break
                    
            except Exception as e:
                error_info = handle_api_error(e, "Gemini AI request", logger, attempt, self.config.max_retries)
                if error_info["retry_recommended"] and attempt < self.config.max_retries:
                    time.sleep(error_info["wait_time"])
                else:
                    logger.error("Max retries exceeded due to unexpected errors")
                    break
        
        return None
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON formatting issues from AI responses.
        
        Args:
            json_str (str): Raw JSON string from AI.
            
        Returns:
            str: Cleaned JSON string.
        """
        import re
        
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix common quote escaping issues
        json_str = json_str.replace('\\"', '"').replace('\\n', '\\n').replace('\\t', '\\t')
        
        # Ensure proper JSON structure
        json_str = json_str.strip()
        
        logger.debug(f"Cleaned JSON: {json_str}")
        return json_str
    
    def _aggressive_json_fix(self, json_str: str) -> Optional[str]:
        """Aggressively attempt to fix malformed JSON from AI responses.
        
        Args:
            json_str (str): Malformed JSON string.
            
        Returns:
            Optional[str]: Fixed JSON string or None if unable to fix.
        """
        import re
        
        try:
            # More aggressive cleaning
            
            # Remove extra commas at the end before closing braces
            json_str = re.sub(r',+(\s*})', r'\1', json_str)
            json_str = re.sub(r',+(\s*\])', r'\1', json_str)
            
            # Fix multiple commas
            json_str = re.sub(r',+', ',', json_str)
            
            # Fix common quote issues
            json_str = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', json_str)
            
            # Try to parse basic structure and rebuild
            # Look for key patterns
            signal_match = re.search(r'"?signal"?\s*:\s*"?(BUY|SELL|HOLD)"?', json_str, re.IGNORECASE)
            confidence_match = re.search(r'"?confidence"?\s*:\s*([0-9.]+)', json_str)
            reasoning_match = re.search(r'"?(reasoning|rationale)"?\s*:\s*"([^"]*)"', json_str)
            
            if signal_match and confidence_match:
                # Rebuild the JSON structure
                signal = signal_match.group(1).upper()
                confidence = float(confidence_match.group(1))
                reasoning = reasoning_match.group(2) if reasoning_match else "AI analysis completed"
                
                fixed_json = {
                    "signal": signal,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
                
                fixed_json_str = json.dumps(fixed_json)
                logger.debug(f"Aggressively fixed JSON: {fixed_json_str}")
                return fixed_json_str
            
        except Exception as e:
            logger.debug(f"Aggressive JSON fix failed: {e}")
            
        return None
    
    def _parse_ai_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse AI response and extract structured data with robust JSON cleaning.
        
        Args:
            response_text (str): Raw AI response text.
            
        Returns:
            Optional[Dict[str, Any]]: Parsed response or None if invalid.
        """
        try:
            # Try to find JSON in the response (in case AI adds extra text)
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                
                # Clean common JSON formatting issues from AI responses
                json_str = self._clean_json_string(json_str)
                
                parsed = json.loads(json_str)
                
                # Validate required fields (support both rationale and reasoning)
                required_fields = ['signal', 'confidence']
                reasoning_field = None
                
                if 'reasoning' in parsed:
                    reasoning_field = 'reasoning'
                elif 'rationale' in parsed:
                    reasoning_field = 'rationale'
                else:
                    logger.error(f"Missing reasoning/rationale field in AI response: {parsed}")
                    return None
                
                if not all(field in parsed for field in required_fields):
                    logger.error(f"Missing required fields in AI response: {parsed}")
                    return None
                
                # Validate signal value
                if parsed['signal'] not in ['BUY', 'SELL', 'HOLD']:
                    logger.error(f"Invalid signal value: {parsed['signal']}")
                    return None
                
                # Validate confidence range
                try:
                    confidence = float(parsed['confidence'])
                    if not 0.0 <= confidence <= 1.0:
                        logger.error(f"Confidence out of range: {confidence}")
                        return None
                    parsed['confidence'] = confidence
                except (ValueError, TypeError):
                    logger.error(f"Invalid confidence value: {parsed['confidence']}")
                    return None
                
                # Validate reasoning/rationale
                reasoning_text = parsed[reasoning_field]
                if not reasoning_text or not isinstance(reasoning_text, str):
                    logger.error(f"Invalid {reasoning_field}: {reasoning_text}")
                    return None
                
                # Normalize the field name to 'rationale' for consistency
                if reasoning_field == 'reasoning':
                    parsed['rationale'] = parsed.pop('reasoning')
                
                return parsed
            else:
                logger.error("No JSON found in AI response")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}")
            
            # Try additional cleaning for malformed JSON
            try:
                # Log the problematic JSON for debugging
                logger.debug(f"Problematic JSON: {json_str}")
                
                # Additional aggressive cleaning
                json_str_fixed = self._aggressive_json_fix(json_str)
                
                if json_str_fixed:
                    parsed = json.loads(json_str_fixed)
                    logger.info("Successfully recovered from malformed JSON")
                    
                    # Continue with validation
                    required_fields = ['signal', 'confidence']
                    reasoning_field = None
                    
                    if 'reasoning' in parsed:
                        reasoning_field = 'reasoning'
                    elif 'rationale' in parsed:
                        reasoning_field = 'rationale'
                    else:
                        logger.error(f"Missing reasoning/rationale field in recovered JSON: {parsed}")
                        return None
                    
                    if not all(field in parsed for field in required_fields):
                        logger.error(f"Missing required fields in recovered JSON: {parsed}")
                        return None
                    
                    # Validate signal value
                    if parsed['signal'] not in ['BUY', 'SELL', 'HOLD']:
                        logger.error(f"Invalid signal value in recovered JSON: {parsed['signal']}")
                        return None
                    
                    # Validate confidence range
                    try:
                        confidence = float(parsed['confidence'])
                        if not 0.0 <= confidence <= 1.0:
                            logger.error(f"Confidence out of range in recovered JSON: {confidence}")
                            return None
                        parsed['confidence'] = confidence
                    except (ValueError, TypeError):
                        logger.error(f"Invalid confidence value in recovered JSON: {parsed['confidence']}")
                        return None
                    
                    # Validate reasoning/rationale
                    reasoning_text = parsed[reasoning_field]
                    if not reasoning_text or not isinstance(reasoning_text, str):
                        logger.error(f"Invalid {reasoning_field} in recovered JSON: {reasoning_text}")
                        return None
                    
                    # Normalize the field name to 'rationale' for consistency
                    if reasoning_field == 'reasoning':
                        parsed['rationale'] = parsed.pop('reasoning')
                    
                    return parsed
                else:
                    logger.error("Failed to recover malformed JSON")
                    return None
                    
            except Exception as recovery_error:
                logger.error(f"JSON recovery attempt failed: {recovery_error}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error parsing AI response: {e}")
            return None
    
    def analyze_news_sentiment(
        self, 
        ticker: str, 
        articles: List[NewsArticle], 
        context: Optional[str] = None
    ) -> Optional[AIAnalysisResult]:
        """Analyze news sentiment for trading signals.
        
        Args:
            ticker (str): Stock ticker symbol.
            articles (List[NewsArticle]): News articles to analyze.
            context (str, optional): Additional context for analysis.
            
        Returns:
            Optional[AIAnalysisResult]: Analysis result or None if failed.
        """
        if not articles:
            logger.warning("No news articles provided for analysis")
            return None
        
        if not self.client:
            logger.error("AI client not initialized - cannot perform analysis")
            return None
        
        # Construct prompt
        prompt = self._construct_analysis_prompt(ticker, articles, context)
        
        # Make AI request
        response_text = self._make_ai_request(prompt)
        if not response_text:
            logger.error("Failed to get response from AI model")
            return None
        
        # Parse response
        parsed_response = self._parse_ai_response(response_text)
        if not parsed_response:
            logger.error("Failed to parse AI response")
            return None
        
        # Create result object
        try:
            result = AIAnalysisResult(
                signal=parsed_response['signal'],
                confidence=parsed_response['confidence'],
                rationale=parsed_response['rationale'],
                model_used=self.model_name,
                analysis_timestamp=time.time(),
                raw_response=response_text
            )
            
            logger.info(f"AI analysis completed: {result.signal} with {result.confidence:.2f} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create analysis result: {e}")
            return None
    
    def analyze_single_text(
        self, 
        ticker: str, 
        text: str, 
        context: Optional[str] = None
    ) -> Optional[AIAnalysisResult]:
        """Analyze a single text for sentiment.
        
        Args:
            ticker (str): Stock ticker symbol.
            text (str): Text to analyze.
            context (str, optional): Additional context.
            
        Returns:
            Optional[AIAnalysisResult]: Analysis result or None if failed.
        """
        # Create a mock article from the text
        import datetime
        mock_article = NewsArticle(
            source="Direct Input",
            title=text[:100],  # Use first 100 chars as title
            description=text,
            url="",
            published_at=datetime.datetime.now(datetime.timezone.utc)
        )
        
        return self.analyze_news_sentiment(ticker, [mock_article], context)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test AI connection and return status.
        
        Returns:
            Dict[str, Any]: Connection test results.
        """
        # Use centralized API key validation
        api_validation = APIValidator.validate_api_key(
            self.config.google_api_key, 
            APIProvider.GOOGLE_GEMINI
        )
        
        status = {
            "connected": False,
            "api_key_valid": api_validation.is_valid,
            "client_initialized": self.client is not None,
            "model_name": self.model_name,
            "error": api_validation.error_message if not api_validation.is_valid else None,
            "test_response": None
        }
        
        if not self.client:
            status["error"] = "Client not initialized"
            return status
        
        try:
            # Make a simple test request
            test_prompt = """Respond with exactly this JSON: {"test": "success", "model": "working"}"""
            
            response = self._make_ai_request(test_prompt)
            if response:
                status["connected"] = True
                status["test_response"] = response
            else:
                status["error"] = "No response from AI model"
                
        except Exception as e:
            status["error"] = f"Connection test failed: {str(e)}"
        
        return status
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics and configuration.
        
        Returns:
            Dict[str, Any]: Analysis statistics and settings.
        """
        return {
            "model_name": self.model_name,
            "temperature": self.config.ai_temperature,
            "max_tokens": self.config.ai_max_tokens,
            "rate_limit_requests": self.config.gemini_rate_limit_requests,
            "max_retries": self.config.max_retries,
            "last_request_time": self.last_request_time,
            "client_initialized": self.client is not None
        }


# Global singleton instance
_ai_analyzer_instance = None
_analyzer_lock = threading.Lock()


def get_ai_analyzer() -> AIAnalyzer:
    """Get global AI analyzer instance (singleton pattern).
    
    Returns:
        AIAnalyzer: Global AI analyzer instance.
    """
    global _ai_analyzer_instance
    
    if _ai_analyzer_instance is None:
        with _analyzer_lock:
            if _ai_analyzer_instance is None:
                _ai_analyzer_instance = AIAnalyzer()
    
    return _ai_analyzer_instance


# Convenience functions for direct usage
def analyze_news_sentiment(
    ticker: str, 
    articles: List[NewsArticle], 
    context: Optional[str] = None
) -> Optional[AIAnalysisResult]:
    """Convenience function for news sentiment analysis.
    
    Args:
        ticker (str): Stock ticker symbol.
        articles (List[NewsArticle]): News articles to analyze.
        context (str, optional): Additional context.
        
    Returns:
        Optional[AIAnalysisResult]: Analysis result or None if failed.
    """
    analyzer = get_ai_analyzer()
    return analyzer.analyze_news_sentiment(ticker, articles, context)


def test_ai_connection() -> Dict[str, Any]:
    """Test AI connection using global analyzer.
    
    Returns:
        Dict[str, Any]: Connection test results.
    """
    analyzer = get_ai_analyzer()
    return analyzer.test_connection()


def validate_analysis_result(result: AIAnalysisResult) -> bool:
    """Validate an AI analysis result.
    
    Args:
        result (AIAnalysisResult): Result to validate.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    if not result:
        return False
    
    # Check signal
    if result.signal not in ['BUY', 'SELL', 'HOLD']:
        return False
    
    # Check confidence
    if not isinstance(result.confidence, (int, float)) or not 0.0 <= result.confidence <= 1.0:
        return False
    
    # Check rationale
    if not result.rationale or not isinstance(result.rationale, str):
        return False
    
    return True


def create_mock_analysis_result(
    signal: str = "HOLD", 
    confidence: float = 0.5, 
    rationale: str = "Mock analysis for testing"
) -> AIAnalysisResult:
    """Create a mock analysis result for testing.
    
    Args:
        signal (str): Signal type.
        confidence (float): Confidence level.
        rationale (str): Analysis rationale.
        
    Returns:
        AIAnalysisResult: Mock analysis result.
    """
    return AIAnalysisResult(
        signal=signal,
        confidence=confidence,
        rationale=rationale,
        model_used="mock",
        analysis_timestamp=time.time(),
        raw_response=f'{{"signal": "{signal}", "confidence": {confidence}, "rationale": "{rationale}"}}'
    )