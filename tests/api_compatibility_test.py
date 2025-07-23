#!/usr/bin/env python3
"""
API Updates Test Script for AI Trading Demo

This script tests the updated AI analyzer and news fetcher with real API responses
to ensure compatibility and functionality after code modifications.

Author: AI Trading Demo Team
Version: 2.0 (Updated API Integration)
"""

import sys
import os
import logging
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import get_config, validate_environment
from core.news_fetcher import get_news_fetcher, validate_news_api
from core.ai_analyzer import get_ai_analyzer, test_ai_connection


def setup_logging() -> None:
    """Configure logging for test output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def test_environment_setup() -> Dict[str, Any]:
    """Test environment variable setup and API key validation."""
    print("\n🔧 Testing Environment Setup...")

    config = get_config()
    validation = validate_environment()

    print(f"✅ API Keys Validation:")
    print(
        f"   Google API Key: {'✅ Valid' if validation['api_key_status']['google_api_key'] else '❌ Invalid'}"
    )
    print(
        f"   NewsAPI Key: {'✅ Valid' if validation['api_key_status']['newsapi_api_key'] else '❌ Invalid'}"
    )
    print(
        f"   Configuration Valid: {'✅ Yes' if validation['configuration_valid'] else '❌ No'}"
    )

    if validation["errors"]:
        print(f"❌ Errors found:")
        for error in validation["errors"]:
            print(f"   - {error}")

    return validation


def test_news_api_updates() -> Dict[str, Any]:
    """Test updated NewsAPI functionality with improved queries."""
    print("\n📰 Testing Updated NewsAPI Functionality...")

    # Test connection validation
    news_status = validate_news_api()
    print(
        f"📡 NewsAPI Connection: {'✅ Connected' if news_status['connected'] else '❌ Failed'}"
    )

    if not news_status["connected"]:
        print(f"❌ Error: {news_status.get('error', 'Unknown')}")
        return news_status

    # Test improved stock news fetching
    print(f"📈 Testing improved stock news queries...")
    fetcher = get_news_fetcher()

    test_tickers = ["AAPL", "TSLA", "UNKNOWN_TICKER"]
    for ticker in test_tickers:
        print(f"\n   Testing {ticker}:")
        try:
            articles = fetcher.get_stock_news(ticker, max_articles=3, use_cache=False)
            print(f"   📄 Found {len(articles)} articles")

            if articles:
                for i, article in enumerate(articles[:2], 1):
                    print(f"      {i}. {article.source}: {article.title[:60]}...")
            else:
                print(f"   ⚠️  No articles found for {ticker}")

        except Exception as e:
            print(f"   ❌ Error fetching {ticker}: {e}")

    return {"status": "completed", "connection": news_status["connected"]}


def test_ai_analyzer_updates() -> Dict[str, Any]:
    """Test updated AI analyzer with new model and prompt structure."""
    print("\n🤖 Testing Updated AI Analyzer...")

    # Test AI connection
    ai_status = test_ai_connection()
    print(
        f"🧠 AI Connection: {'✅ Connected' if ai_status['connected'] else '❌ Failed'}"
    )

    if not ai_status["connected"]:
        print(f"❌ Error: {ai_status.get('error', 'Unknown')}")
        return ai_status

    print(f"🎯 Model: {ai_status.get('model_name', 'Unknown')}")

    # Test news analysis with real data
    print(f"\n🔍 Testing AI analysis with real news data...")

    try:
        fetcher = get_news_fetcher()
        analyzer = get_ai_analyzer()

        # Get some real news for AAPL
        articles = fetcher.get_stock_news("AAPL", max_articles=3, use_cache=False)

        if articles:
            print(f"📄 Analyzing {len(articles)} AAPL articles...")
            result = analyzer.analyze_news_sentiment("AAPL", articles)

            if result:
                print(f"✅ Analysis Result:")
                print(f"   Signal: {result.signal}")
                print(f"   Confidence: {result.confidence:.2f}")
                print(f"   Reasoning: {result.rationale[:100]}...")
                print(f"   Model Used: {result.model_used}")

                # Test numerical signal conversion
                numerical_signal = result.get_numerical_signal()
                print(f"   Numerical Signal: {numerical_signal}")

                return {
                    "status": "success",
                    "signal": result.signal,
                    "confidence": result.confidence,
                    "model": result.model_used,
                }
            else:
                print("❌ Failed to get analysis result")
                return {"status": "failed", "error": "No analysis result"}
        else:
            print("⚠️  No articles available for testing")
            return {"status": "no_data", "error": "No articles found"}

    except Exception as e:
        print(f"❌ AI Analysis Error: {e}")
        return {"status": "error", "error": str(e)}


def test_json_response_parsing() -> Dict[str, Any]:
    """Test AI response parsing with different JSON formats."""
    print("\n🔧 Testing JSON Response Parsing...")

    analyzer = get_ai_analyzer()

    # Test different response formats
    test_responses = [
        # Standard format with 'reasoning'
        '{"signal": "BUY", "confidence": 0.85, "reasoning": "Strong earnings performance"}',
        # Format with 'rationale' (legacy)
        '{"signal": "HOLD", "confidence": 0.60, "rationale": "Mixed market signals"}',
        # Format with extra text around JSON
        'Based on my analysis:\n{"signal": "SELL", "confidence": 0.75, "reasoning": "Declining revenue trends"}\nThis is my recommendation.',
        # Invalid format (missing field)
        '{"signal": "BUY", "confidence": 0.90}',
        # Invalid signal value
        '{"signal": "MAYBE", "confidence": 0.50, "reasoning": "Uncertain outlook"}',
    ]

    results = []
    for i, response_text in enumerate(test_responses, 1):
        print(f"\n   Test {i}: ", end="")
        try:
            parsed = analyzer._parse_ai_response(response_text)
            if parsed:
                print(
                    f"✅ Success - Signal: {parsed['signal']}, Confidence: {parsed['confidence']}"
                )
                results.append({"test": i, "status": "success", "parsed": parsed})
            else:
                print(f"❌ Failed to parse")
                results.append({"test": i, "status": "failed"})
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({"test": i, "status": "error", "error": str(e)})

    successful_tests = sum(1 for r in results if r["status"] == "success")
    print(
        f"\n📊 JSON Parsing Results: {successful_tests}/{len(test_responses)} tests passed"
    )

    return {"successful_tests": successful_tests, "total_tests": len(test_responses)}


def test_cache_performance() -> Dict[str, Any]:
    """Test caching performance and functionality."""
    print("\n⚡ Testing Cache Performance...")

    import time

    fetcher = get_news_fetcher()

    # Clear cache first
    fetcher.clear_cache()

    # First request (no cache)
    start_time = time.time()
    articles1 = fetcher.get_stock_news("AAPL", max_articles=5, use_cache=True)
    first_request_time = time.time() - start_time

    # Second request (with cache)
    start_time = time.time()
    articles2 = fetcher.get_stock_news("AAPL", max_articles=5, use_cache=True)
    second_request_time = time.time() - start_time

    print(f"📈 Cache Performance:")
    print(
        f"   First request (no cache): {first_request_time:.2f}s - {len(articles1)} articles"
    )
    print(
        f"   Second request (cached): {second_request_time:.2f}s - {len(articles2)} articles"
    )

    if second_request_time < first_request_time:
        speedup = (
            first_request_time / second_request_time
            if second_request_time > 0
            else float("inf")
        )
        print(f"   ✅ Cache speedup: {speedup:.1f}x faster")
    else:
        print(f"   ⚠️  Cache may not be working effectively")

    # Test cache stats
    cache_stats = fetcher.get_cache_stats()
    print(f"   📊 Cache entries: {cache_stats['entries']}")

    return {
        "first_request_time": first_request_time,
        "second_request_time": second_request_time,
        "cache_entries": cache_stats["entries"],
    }


def main():
    """Run all API update tests."""
    print("🚀 AI Trading Demo - API Updates Test Suite")
    print("=" * 50)

    setup_logging()

    test_results = {}

    # Test environment setup
    test_results["environment"] = test_environment_setup()

    if not test_results["environment"]["api_keys_valid"]:
        print("\n❌ Cannot proceed with API tests - API keys not configured properly")
        print("Please check your .env file and ensure API keys are set correctly")
        return False

    # Test news API updates
    test_results["news_api"] = test_news_api_updates()

    # Test AI analyzer updates
    test_results["ai_analyzer"] = test_ai_analyzer_updates()

    # Test JSON parsing
    test_results["json_parsing"] = test_json_response_parsing()

    # Test cache performance
    test_results["cache_performance"] = test_cache_performance()

    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")

    environment_ok = test_results["environment"]["api_keys_valid"]
    news_ok = test_results["news_api"].get("connection", False)
    ai_ok = test_results["ai_analyzer"].get("status") == "success"
    json_ok = test_results["json_parsing"]["successful_tests"] >= 3
    cache_ok = test_results["cache_performance"]["cache_entries"] > 0

    print(f"   Environment Setup: {'✅ Pass' if environment_ok else '❌ Fail'}")
    print(f"   NewsAPI Integration: {'✅ Pass' if news_ok else '❌ Fail'}")
    print(f"   AI Analyzer: {'✅ Pass' if ai_ok else '❌ Fail'}")
    print(f"   JSON Parsing: {'✅ Pass' if json_ok else '❌ Fail'}")
    print(f"   Cache Performance: {'✅ Pass' if cache_ok else '❌ Fail'}")

    all_tests_passed = all([environment_ok, news_ok, ai_ok, json_ok, cache_ok])

    if all_tests_passed:
        print(f"\n🎉 All tests passed! API updates are working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Please review the results above.")

    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
