#!/usr/bin/env python3
"""
Integration Test Script for AI Trading Demo

Tests the complete pipeline with real API calls:
- NewsAPI fetching
- Google Gemini AI analysis
- Signal generation
- End-to-end workflow

Author: AI Trading Demo Team
Version: 2.0 (AI-Powered Architecture)
"""

import os
import sys
import datetime
import pandas as pd
from typing import Optional

# API keys should be set as environment variables before running this test
# Example:
# export GOOGLE_API_KEY=your_key_here
# export NEWSAPI_API_KEY=your_key_here

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from core.config import get_config, validate_environment
from core.news_fetcher import get_news_fetcher, fetch_stock_news
from core.ai_analyzer import get_ai_analyzer, test_ai_connection
from core.data_manager import DataManager
from core.strategy import generate_trading_signals_with_ui_feedback


def test_environment_setup():
    """Test environment configuration and API key availability."""
    print("🔧 Testing Environment Setup...")

    config = get_config()
    validation = validate_environment()

    print(
        f"   Google API Key: {'✅ Available' if validation['api_key_status']['google_api_key'] else '❌ Missing'}"
    )
    print(
        f"   NewsAPI Key: {'✅ Available' if validation['api_key_status']['newsapi_api_key'] else '❌ Missing'}"
    )
    print(f"   AI Model: {config.ai_model_name}")

    if not validation["api_keys_valid"]:
        print("❌ API Keys not properly configured!")
        for error in validation["errors"]:
            print(f"   Error: {error}")
        return False

    print("✅ Environment setup is valid\n")
    return True


def test_news_fetching():
    """Test news fetching functionality."""
    print("📰 Testing News Fetching...")

    try:
        news_fetcher = get_news_fetcher()

        # Test connection first
        print("   Testing NewsAPI connection...")
        connection_status = news_fetcher.validate_connection()
        if not connection_status.get("connected", False):
            print(
                f"   ❌ NewsAPI connection failed: {connection_status.get('error', 'Unknown error')}"
            )
            return None

        print("   ✅ NewsAPI connection successful")

        # Fetch actual news
        print("   Fetching AAPL news...")
        articles = fetch_stock_news("AAPL", max_articles=5)

        if not articles:
            print("   ❌ No news articles fetched")
            return None

        print(f"   ✅ Fetched {len(articles)} news articles")
        for i, article in enumerate(articles[:3], 1):
            print(f"   {i}. {article.title[:60]}...")

        return articles

    except Exception as e:
        print(f"   ❌ News fetching error: {e}")
        return None


def test_ai_analysis(articles):
    """Test AI analysis functionality."""
    print("\n🤖 Testing AI Analysis...")

    if not articles:
        print("   ❌ No articles to analyze")
        return None

    try:
        # Test AI connection
        print("   Testing Gemini AI connection...")
        connection_status = test_ai_connection()

        if not connection_status["connected"]:
            print(
                f"   ❌ Gemini AI connection failed: {connection_status.get('error', 'Unknown error')}"
            )
            return None

        print("   ✅ Gemini AI connection successful")
        print(f"   Model: {connection_status.get('model_name', 'Unknown')}")

        # Perform AI analysis
        print("   Analyzing news sentiment...")
        ai_analyzer = get_ai_analyzer()
        result = ai_analyzer.analyze_news_sentiment("AAPL", articles)

        if not result:
            print("   ❌ AI analysis failed")
            return None

        print(f"   ✅ AI Analysis completed")
        print(f"   Signal: {result.signal}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Rationale: {result.rationale[:100]}...")

        return result

    except Exception as e:
        print(f"   ❌ AI analysis error: {e}")
        return None


def test_stock_data_loading():
    """Test stock data loading."""
    print("\n📊 Testing Stock Data Loading...")

    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=30)

        print(f"   Loading AAPL data from {start_date} to {end_date}...")
        data_manager = DataManager()
        stock_data = data_manager.fetch_stock_data("AAPL", start_date, end_date)

        if stock_data.empty:
            print("   ❌ No stock data loaded")
            return None

        print(f"   ✅ Loaded {len(stock_data)} days of stock data")
        print(
            f"   Price range: ${stock_data['Close'].min():.2f} - ${stock_data['Close'].max():.2f}"
        )

        return stock_data

    except Exception as e:
        print(f"   ❌ Stock data loading error: {e}")
        return None


def test_signal_generation(stock_data):
    """Test complete signal generation."""
    print("\n🎯 Testing Signal Generation...")

    if stock_data is None or stock_data.empty:
        print("   ❌ No stock data available for signal generation")
        return None

    try:
        print("   Generating AI-powered trading signals...")
        result_data = generate_trading_signals_with_ui_feedback(stock_data, "AAPL")

        if result_data is None or result_data.empty:
            print("   ❌ Signal generation failed")
            return None

        # Analyze results
        signals = result_data[result_data["Signal"] != 0]

        print(f"   ✅ Signal generation completed")
        print(f"   Total signals generated: {len(signals)}")

        if not signals.empty:
            buy_signals = len(signals[signals["Signal"] == 1])
            sell_signals = len(signals[signals["Signal"] == -1])
            print(f"   Buy signals: {buy_signals}")
            print(f"   Sell signals: {sell_signals}")

            if "AI_Confidence" in signals.columns:
                avg_confidence = signals["AI_Confidence"].mean()
                print(f"   Average AI confidence: {avg_confidence:.1%}")

        return result_data

    except Exception as e:
        print(f"   ❌ Signal generation error: {e}")
        return None


def main():
    """Run complete integration test."""
    print("🚀 AI Trading Demo - Integration Test")
    print("=" * 50)

    # Test environment setup
    if not test_environment_setup():
        print("\n❌ Integration test failed due to environment setup issues")
        sys.exit(1)

    # Test news fetching
    articles = test_news_fetching()

    # Test AI analysis
    ai_result = test_ai_analysis(articles)

    # Test stock data loading
    stock_data = test_stock_data_loading()

    # Test complete signal generation
    final_data = test_signal_generation(stock_data)

    # Final assessment
    print("\n" + "=" * 50)
    print("🏁 Integration Test Results:")
    print(f"   Environment Setup: {'✅' if True else '❌'}")
    print(f"   News Fetching: {'✅' if articles else '❌'}")
    print(f"   AI Analysis: {'✅' if ai_result else '❌'}")
    print(f"   Stock Data Loading: {'✅' if stock_data is not None else '❌'}")
    print(f"   Signal Generation: {'✅' if final_data is not None else '❌'}")

    success = all(
        [
            articles is not None,
            ai_result is not None,
            stock_data is not None,
            final_data is not None,
        ]
    )

    if success:
        print(
            "\n🎉 All integration tests passed! The AI trading system is working correctly."
        )

        # Show a sample of the final results
        if final_data is not None and not final_data.empty:
            recent_signals = final_data[final_data["Signal"] != 0].tail(3)
            if not recent_signals.empty:
                print("\n📊 Recent AI Trading Signals:")
                for _, row in recent_signals.iterrows():
                    signal_type = "BUY" if row["Signal"] == 1 else "SELL"
                    confidence = row.get("AI_Confidence", 0)
                    print(
                        f"   {row['Date'].strftime('%Y-%m-%d')}: {signal_type} at ${row['Close']:.2f} (Confidence: {confidence:.1%})"
                    )
    else:
        print(
            "\n❌ Some integration tests failed. Please check the error messages above."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
