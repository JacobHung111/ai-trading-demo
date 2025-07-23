"""
Trading Strategy Module for AI Trading Demo

This module implements AI-powered trading strategy logic shared between 
Streamlit and NiceGUI applications. The strategy generates buy/sell signals 
based on AI sentiment analysis of news headlines using Google Gemini API.

Author: AI Trading Demo Team
Version: 2.0 (AI-Powered Hybrid Architecture)
"""

import datetime
import time
import logging
import pandas as pd
from typing import Dict, List, Optional, Any

from .config import get_config
from .news_fetcher import get_news_fetcher, NewsArticle
from .ai_analyzer import get_ai_analyzer
from ui.components import (
    display_streamlit_message, MessageType
)


class TradingStrategy:
    """AI-powered trading strategy implementation."""

    def __init__(self, config=None):
        """Initialize the AI trading strategy with configuration.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.news_fetcher = get_news_fetcher()
        self.ai_analyzer = get_ai_analyzer()
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters from config
        self.max_articles = self.config.news_articles_count
        self.max_age_hours = self.config.max_news_age_hours
        
        # Cache for analysis results to avoid duplicate API calls
        self._analysis_cache = {}
        self._cache_ttl = 3600  # 1 hour cache

    def generate_signals(self, data: pd.DataFrame, ticker: str = "AAPL") -> pd.DataFrame:
        """Generate trading signals based on AI-powered news sentiment analysis.

        OPTIMIZED APPROACH:
        1. Fetch ALL news for the entire date range in ONE API call
        2. Group news by date
        3. Batch analyze each day with news using AI
        4. Apply results to respective days

        Args:
            data (pd.DataFrame): OHLCV stock price data with Date column or DatetimeIndex.
            ticker (str): Stock ticker symbol for news analysis.

        Returns:
            pd.DataFrame: Original data with added 'Signal', 'AI_Signal', 
                         'AI_Confidence', and 'AI_Rationale' columns.
                         Only days with news will have analysis results.
        """
        try:
            # Create a copy to avoid modifying original data
            result_data = data.copy()

            # Initialize AI analysis columns with default values
            result_data["Signal"] = 0
            result_data["AI_Signal"] = "No analysis performed"
            result_data["AI_Confidence"] = 0.0
            result_data["AI_Rationale"] = "No analysis performed"

            self.logger.info(f"Generating optimized AI signals for {ticker} with {len(data)} data points")

            # Ensure we have a Date column for iteration
            if 'Date' not in result_data.columns:
                if isinstance(result_data.index, pd.DatetimeIndex):
                    result_data['Date'] = result_data.index.date
                else:
                    self.logger.error("No Date column found and index is not DatetimeIndex")
                    return result_data

            # Get date range for analysis
            start_date = result_data['Date'].min() 
            end_date = result_data['Date'].max()
            self.logger.info(f"Analyzing date range: {start_date} to {end_date}")

            # STEP 1: Fetch ALL news for the entire date range in ONE API call
            self.logger.info(f"Fetching all news for {ticker} from {start_date} to {end_date}")
            all_articles = self._fetch_range_news(ticker, start_date, end_date)
            
            if not all_articles:
                self.logger.warning(f"No news articles found for {ticker} in the date range")
                return result_data

            # STEP 2: Group news by date
            news_by_date = self._group_news_by_date(all_articles)
            self.logger.info(f"Found news for {len(news_by_date)} different dates")

            # STEP 3: Batch analyze each day with news
            days_analyzed = 0
            for analysis_date, date_articles in news_by_date.items():
                cache_key = f"{ticker}_{analysis_date.strftime('%Y%m%d')}"
                
                analysis_result = None
                
                # Check cache first
                if cache_key in self._analysis_cache:
                    analysis_result = self._analysis_cache[cache_key]
                    self.logger.debug(f"Using cached analysis for {ticker} on {analysis_date}")
                else:
                    # AI Analysis for this day's news
                    self.logger.debug(f"Analyzing sentiment for {ticker} on {analysis_date} with {len(date_articles)} articles")
                    analysis_result = self.ai_analyzer.analyze_news_sentiment(
                        ticker=ticker,
                        articles=date_articles
                    )

                    if not analysis_result:
                        self.logger.warning(f"AI analysis failed for {ticker} on {analysis_date}")
                        continue

                    # Cache the result
                    self._analysis_cache[cache_key] = analysis_result
                    self.logger.debug(f"Cached analysis result for {ticker} on {analysis_date}")

                # STEP 4: Apply results to the correct day in our data
                if analysis_result:
                    # Find the row(s) for this date
                    matching_rows = result_data[result_data['Date'] == analysis_date]
                    
                    if not matching_rows.empty:
                        for idx in matching_rows.index:
                            signal_value = analysis_result.get_numerical_signal()
                            
                            result_data.loc[idx, "Signal"] = signal_value
                            result_data.loc[idx, "AI_Signal"] = analysis_result.signal
                            result_data.loc[idx, "AI_Confidence"] = analysis_result.confidence
                            result_data.loc[idx, "AI_Rationale"] = analysis_result.rationale
                            
                        days_analyzed += 1
                        self.logger.info(
                            f"Day {days_analyzed}: {analysis_date} - {analysis_result.signal} "
                            f"(confidence: {analysis_result.confidence:.2f})"
                        )

            self.logger.info(f"Optimized analysis complete: {days_analyzed} days analyzed using {len(all_articles)} total articles")
            return result_data

        except Exception as e:
            self.logger.error(f"Error generating optimized AI trading signals: {e}")
            return pd.DataFrame()

    def _fetch_range_news(self, ticker: str, start_date: datetime.date, end_date: datetime.date) -> List[NewsArticle]:
        """Fetch ALL news for the entire date range in one optimized API call.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (datetime.date): Start date of range.
            end_date (datetime.date): End date of range.
            
        Returns:
            List[NewsArticle]: All news articles for the date range.
        """
        try:
            # Calculate hours between start and end date
            date_diff = (end_date - start_date).days
            max_age_hours = min(date_diff * 24 + 48, 720)  # Cap at 30 days (720 hours)
            
            # Fetch all news for the range
            articles = self.news_fetcher.get_stock_news(
                ticker=ticker,
                max_articles=self.max_articles * 3,  # Get more articles to cover all dates
                max_age_hours=max_age_hours,
                use_cache=True
            )
            
            # Filter articles to only include those within our date range
            filtered_articles = []
            for article in articles:
                article_date = article.published_at.date()
                if start_date <= article_date <= end_date:
                    filtered_articles.append(article)
            
            self.logger.info(f"Fetched {len(filtered_articles)} articles within date range from {len(articles)} total")
            return filtered_articles
            
        except Exception as e:
            self.logger.error(f"Error fetching range news for {ticker}: {e}")
            return []

    def _group_news_by_date(self, articles: List[NewsArticle]) -> Dict[datetime.date, List[NewsArticle]]:
        """Group news articles by their published date.
        
        Args:
            articles (List[NewsArticle]): List of news articles.
            
        Returns:
            Dict[datetime.date, List[NewsArticle]]: Articles grouped by date.
        """
        news_by_date = {}
        
        for article in articles:
            article_date = article.published_at.date()
            
            if article_date not in news_by_date:
                news_by_date[article_date] = []
            
            news_by_date[article_date].append(article)
        
        # Sort articles within each date by time (most recent first)
        for date_key in news_by_date:
            news_by_date[date_key].sort(key=lambda x: x.published_at, reverse=True)
        
        return news_by_date


    def get_latest_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Get the most recent AI trading signal and its details.

        This function is useful for real-time applications that need to
        know the current AI signal status.

        Args:
            data (pd.DataFrame): Data with AI signals generated.

        Returns:
            Optional[Dict]: Dictionary containing:
                           {'signal': int, 'date': str, 'price': float, 'type': str,
                            'ai_signal': str, 'confidence': float, 'rationale': str}
                           Returns None if no valid signal found.
        """
        try:
            if data.empty or "Signal" not in data.columns:
                return None

            # Find the most recent non-zero signal
            signals = data[data["Signal"] != 0]
            if signals.empty:
                # Check for most recent analysis even if signal is HOLD
                if "AI_Signal" in data.columns:
                    latest_data = data.iloc[-1]
                    return {
                        "signal": int(latest_data.get("Signal", 0)),
                        "date": latest_data.get("Date", latest_data.name),
                        "price": float(latest_data["Close"]),
                        "type": latest_data.get("AI_Signal", "HOLD"),
                        "ai_signal": latest_data.get("AI_Signal", "HOLD"),
                        "confidence": float(latest_data.get("AI_Confidence", 0.0)),
                        "rationale": latest_data.get("AI_Rationale", "No analysis available"),
                    }
                return None

            latest_signal = signals.iloc[-1]

            signal_type = "BUY" if latest_signal["Signal"] == 1 else "SELL"

            return {
                "signal": int(latest_signal["Signal"]),
                "date": latest_signal.get("Date", latest_signal.name),
                "price": float(latest_signal["Close"]),
                "type": signal_type,
                "ai_signal": latest_signal.get("AI_Signal", signal_type),
                "confidence": float(latest_signal.get("AI_Confidence", 0.0)),
                "rationale": latest_signal.get("AI_Rationale", "No rationale available"),
            }

        except Exception as e:
            self.logger.error(f"Error getting latest signal: {e}")
            return None

    def get_signal_summary(self, data: pd.DataFrame) -> Optional[Dict]:
        """Generate a summary of all AI trading signals in the dataset.

        This function provides statistical analysis of the AI strategy performance
        useful for both applications' reporting features.

        Args:
            data (pd.DataFrame): Data with AI signals generated.

        Returns:
            Optional[Dict]: Dictionary containing signal statistics:
                           {'total_signals', 'buy_signals', 'sell_signals',
                            'total_days', 'signal_rate', 'avg_confidence', 'analyses_performed'}
                           Returns None if calculation fails.
        """
        try:
            if data.empty or "Signal" not in data.columns:
                return None

            total_days = len(data)
            total_signals = len(data[data["Signal"] != 0])
            buy_signals = len(data[data["Signal"] == 1])
            sell_signals = len(data[data["Signal"] == -1])
            hold_signals = len(data[data["Signal"] == 0])

            signal_rate = (total_signals / total_days * 100) if total_days > 0 else 0

            # AI-specific statistics
            avg_confidence = 0.0
            analyses_performed = 0
            
            if "AI_Confidence" in data.columns:
                # Count rows where AI analysis was performed (confidence > 0)
                ai_data = data[data["AI_Confidence"] > 0]
                analyses_performed = len(ai_data)
                
                if analyses_performed > 0:
                    avg_confidence = ai_data["AI_Confidence"].mean()

            return {
                "total_signals": total_signals,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "hold_signals": hold_signals,
                "total_days": total_days,
                "signal_rate": round(signal_rate, 2),
                "avg_confidence": round(avg_confidence, 3),
                "analyses_performed": analyses_performed,
            }

        except Exception as e:
            self.logger.error(f"Error calculating signal summary: {e}")
            return None

    def get_signal_list(self, data: pd.DataFrame) -> List[Dict]:
        """Get a list of all AI trading signals with their details.

        This function returns all signals in chronological order, useful
        for displaying signal history in both applications.

        Args:
            data (pd.DataFrame): Data with AI signals generated.

        Returns:
            List[Dict]: List of signal dictionaries, each containing:
                       {'date', 'signal', 'type', 'price', 'ai_signal', 
                        'confidence', 'rationale', 'volume'}
                       Returns empty list if no signals found.
        """
        try:
            if data.empty or "Signal" not in data.columns:
                return []

            # Filter to only signal rows (non-zero signals)
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
                        "ai_signal": row.get("AI_Signal", signal_type),
                        "confidence": float(row.get("AI_Confidence", 0.0)),
                        "rationale": row.get("AI_Rationale", "No rationale available"),
                        "volume": int(row.get("Volume", 0)),
                    }
                )

            return signal_list

        except Exception as e:
            self.logger.error(f"Error getting signal list: {e}")
            return []

    def analyze_single_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Perform real-time AI analysis for a single ticker.

        This function is optimized for real-time signal detection and
        can be called independently for live analysis.

        Args:
            ticker (str): Stock ticker symbol to analyze.

        Returns:
            Optional[Dict[str, Any]]: Analysis result containing:
                {'signal': int, 'ai_signal': str, 'confidence': float, 
                 'rationale': str, 'timestamp': float}
                Returns None if analysis fails.
        """
        try:
            self.logger.info(f"Performing real-time analysis for {ticker}")

            # Fetch recent news
            articles = self.news_fetcher.get_stock_news(
                ticker=ticker,
                max_articles=self.max_articles,
                max_age_hours=self.max_age_hours
            )

            if not articles:
                self.logger.warning(f"No news articles found for {ticker}")
                return {
                    "signal": 0,
                    "ai_signal": "HOLD",
                    "confidence": 0.0,
                    "rationale": "No recent news available for analysis",
                    "timestamp": time.time(),
                }

            # Perform AI analysis
            analysis_result = self.ai_analyzer.analyze_news_sentiment(
                ticker=ticker,
                articles=articles
            )

            if not analysis_result:
                self.logger.warning(f"AI analysis failed for {ticker}")
                return None

            return {
                "signal": analysis_result.get_numerical_signal(),
                "ai_signal": analysis_result.signal,
                "confidence": analysis_result.confidence,
                "rationale": analysis_result.rationale,
                "timestamp": analysis_result.analysis_timestamp,
            }

        except Exception as e:
            self.logger.error(f"Error in real-time analysis for {ticker}: {e}")
            return None

    def get_strategy_parameters(self) -> Dict:
        """Get current AI strategy parameters.

        Returns:
            Dict: Strategy configuration including AI analysis parameters.
        """
        return {
            "max_articles": self.max_articles,
            "max_age_hours": self.max_age_hours,
            "ai_model": self.config.ai_model_name,
            "ai_temperature": self.config.ai_temperature,
            "confidence_threshold": self.config.min_confidence_threshold,
            "strategy_name": "AI News Sentiment Analysis",
            "version": "2.0",
        }

    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._analysis_cache.clear()
        self.logger.info("Analysis cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict containing cache information.
        """
        return {
            "cache_entries": len(self._analysis_cache),
            "cache_ttl": self._cache_ttl,
            "cache_keys": list(self._analysis_cache.keys()) if self._analysis_cache else []
        }

    def validate_setup(self) -> Dict[str, Any]:
        """Validate that all components are properly configured.
        
        Returns:
            Dict containing validation results.
        """
        validation = {
            "strategy_ready": True,
            "news_fetcher_ready": False,
            "ai_analyzer_ready": False,
            "config_valid": False,
            "errors": [],
            "warnings": []
        }

        try:
            # Validate configuration
            api_keys = self.config.validate_api_keys()
            if not all(api_keys.values()):
                validation["errors"].append("Missing API keys")
                validation["strategy_ready"] = False
            else:
                validation["config_valid"] = True

            # Validate news fetcher
            news_status = self.news_fetcher.validate_connection()
            validation["news_fetcher_ready"] = news_status.get("connected", False)
            if not validation["news_fetcher_ready"]:
                validation["errors"].append(f"News fetcher not ready: {news_status.get('error', 'Unknown error')}")
                validation["strategy_ready"] = False

            # Validate AI analyzer  
            ai_status = self.ai_analyzer.test_connection()
            validation["ai_analyzer_ready"] = ai_status.get("connected", False)
            if not validation["ai_analyzer_ready"]:
                validation["errors"].append(f"AI analyzer not ready: {ai_status.get('error', 'Unknown error')}")
                validation["strategy_ready"] = False

        except Exception as e:
            validation["errors"].append(f"Validation error: {str(e)}")
            validation["strategy_ready"] = False

        return validation


# Additional utility functions for UI compatibility
def generate_trading_signals_with_ui_feedback(df: pd.DataFrame, ticker: str = "AAPL") -> pd.DataFrame:
    """Generates AI-powered trading signals with UI feedback.

    This function implements the AI news sentiment analysis logic specified
    in the project blueprint with Streamlit UI feedback for better user experience.

    Args:
        df (pd.DataFrame): Input DataFrame containing OHLCV stock price data.
        ticker (str): Stock ticker symbol for news analysis.

    Returns:
        pd.DataFrame: DataFrame with added AI signal columns:
                      'Signal' (1, -1, 0), 'AI_Signal', 'AI_Confidence', 'AI_Rationale'
                      Returns the original DataFrame if analysis fails.
    """
    try:
        if df.empty:
            try:
                import streamlit as st
                display_streamlit_message(
                    MessageType.WARNING,
                    "Invalid Data",
                    "Empty DataFrame provided for AI signal generation.",
                    "⚠️"
                )
            except ImportError:
                print("Warning: Empty DataFrame provided for AI signal generation.")
            return df

        # Use the main AI TradingStrategy class for signal generation
        strategy = TradingStrategy()
        
        try:
            import streamlit as st
            with st.spinner(f"🤖 Analyzing news sentiment for {ticker}..."):
                result_df = strategy.generate_signals(df, ticker=ticker)
            
            # Show analysis results
            if "AI_Confidence" in result_df.columns:
                # Check if ANY analysis was performed by looking at the max confidence
                max_confidence = result_df["AI_Confidence"].max()
                
                if max_confidence > 0:
                    # Find the latest signal for a more relevant message
                    latest_analysis = result_df[result_df["AI_Confidence"] > 0].iloc[-1]
                    latest_signal = latest_analysis["AI_Signal"]
                    latest_confidence = latest_analysis["AI_Confidence"]
                    
                    display_streamlit_message(
                        MessageType.SUCCESS,
                        "AI Analysis Complete",
                        f"Latest Signal: {latest_signal} (Confidence: {latest_confidence:.2f})",
                        "🤖"
                    )
                else:
                    display_streamlit_message(
                        MessageType.INFO,
                        "No News Available",
                        "No recent news found for analysis in the selected date range.",
                        "ℹ️"
                    )
            
            return result_df
            
        except ImportError:
            # Fallback without Streamlit UI
            print(f"Generating AI signals for {ticker}...")
            return strategy.generate_signals(df, ticker=ticker)

    except Exception as e:
        try:
            import streamlit as st
            error_message = str(e)
            
            # Check for specific API quota errors
            if "RESOURCE_EXHAUSTED" in error_message or "quota" in error_message.lower():
                display_streamlit_message(
                    MessageType.ERROR,
                    "API Quota Exhausted",
                    "AI Analysis quota limit reached. Please try again tomorrow or upgrade your plan.",
                    "🚫"
                )
                st.markdown("""
                **Gemini AI free tier quota limit reached (200 requests per day)**
                
                📋 **Solutions**:
                1. ⏰ **Wait for Reset**: Quota resets daily at UTC midnight
                2. 💡 **View Demo Data**: App will display simulated trading signals for reference
                3. 🔑 **Upgrade Plan**: Consider upgrading to paid plan for higher quota
                
                📖 **More Information**: [Gemini API Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits)
                """)
                
                # Return data with demo signals for showcase
                return create_demo_signals(df, ticker)
                
            elif "rate limit" in error_message.lower():
                display_streamlit_message(
                    MessageType.WARNING,
                    "Rate Limit Hit",
                    "AI Analysis rate limit reached. Please wait 10 seconds before trying again.",
                    "⏳"
                )
                return df
            else:
                # Generic error handling
                display_streamlit_message(
                    MessageType.ERROR,
                    "AI Analysis Error",
                    f"Error during AI analysis: {error_message}",
                    "❌"
                )
                return df
                
        except ImportError:
            # Console fallback for non-Streamlit environments
            if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                print("❌ API quota exhausted. Please try again tomorrow or upgrade to paid plan.")
            else:
                print(f"❌ AI analysis error: {str(e)}")
        return df


def create_demo_signals(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Create demo trading signals when API quota is exhausted.
    
    This function provides realistic demo data to showcase the application
    functionality when the AI API is not available.
    
    Args:
        df (pd.DataFrame): Input OHLCV data.
        ticker (str): Stock ticker symbol.
        
    Returns:
        pd.DataFrame: Data with demo AI signals.
    """
    import random
    
    result_df = df.copy()
    
    # Initialize AI analysis columns with default values
    result_df["Signal"] = 0
    result_df["AI_Signal"] = "HOLD"
    result_df["AI_Confidence"] = 0.0
    result_df["AI_Rationale"] = "Demo mode - API quota exhausted"
    
    # Generate realistic demo signals for the last 10% of data
    num_signals = max(1, len(df) // 10)
    signal_indices = random.sample(range(len(df) - num_signals, len(df)), min(num_signals, 5))
    
    demo_signals = [
        ("BUY", 1, 0.75, f"Demo analysis: Positive sentiment detected for {ticker} based on simulated news analysis."),
        ("SELL", -1, 0.68, f"Demo analysis: Bearish indicators found for {ticker} in simulated market sentiment."),
        ("HOLD", 0, 0.55, f"Demo analysis: Mixed signals for {ticker}, maintaining neutral position."),
    ]
    
    for i, idx in enumerate(signal_indices):
        if i < len(demo_signals):
            signal_text, signal_value, confidence, rationale = demo_signals[i]
            result_df.loc[result_df.index[idx], "Signal"] = signal_value
            result_df.loc[result_df.index[idx], "AI_Signal"] = signal_text
            result_df.loc[result_df.index[idx], "AI_Confidence"] = confidence
            result_df.loc[result_df.index[idx], "AI_Rationale"] = rationale
    
    try:
        import streamlit as st
        display_streamlit_message(
            MessageType.INFO,
            "Demo Mode Active",
            "API quota exhausted. Displaying simulated trading signals for demonstration.",
            "🎭"
        )
    except ImportError:
        print("✅ Demo mode: Displaying simulated trading signals")
    
    return result_df


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
        
        # Add AI-specific statistics if available
        if "AI_Confidence" in df.columns:
            ai_data = df[df["Signal"] != 0]
            if not ai_data.empty:
                summary["analyses_performed"] = len(ai_data)
                summary["avg_confidence"] = ai_data["AI_Confidence"].mean()
            else:
                summary["analyses_performed"] = 0
                summary["avg_confidence"] = 0.0
        else:
            summary["analyses_performed"] = 0
            summary["avg_confidence"] = 0.0

        return summary

    except Exception as e:
        try:
            import streamlit as st

            display_streamlit_message(
                MessageType.ERROR,
                "Signal Summary Error", 
                f"Error generating signal summary: {str(e)}",
                "❌"
            )
        except ImportError:
            print(f"Error generating signal summary: {str(e)}")
        return {}
