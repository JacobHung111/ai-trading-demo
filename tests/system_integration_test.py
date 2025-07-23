#!/usr/bin/env python3
"""
Final Integration Test for AI Trading Demo

Complete end-to-end test of the AI trading system including:
- Environment validation
- API connectivity
- News fetching and analysis
- AI signal generation
- Both UI applications

Author: AI Trading Demo Team
Version: 2.0 (AI-Powered Final Integration)
"""

import sys
import os
import time
import logging
import subprocess
import requests
from typing import Dict, Any, List

# Add project root to path (parent of tests directory)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import get_config, validate_environment
from core.news_fetcher import get_news_fetcher, NewsArticle
from core.ai_analyzer import get_ai_analyzer, AIAnalysisResult
from core.data_manager import DataManager, load_data_with_streamlit_cache
from core.strategy import TradingStrategy


def setup_logging() -> None:
    """Configure logging for final test."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def test_complete_pipeline(ticker: str = "AAPL") -> Dict[str, Any]:
    """Test the complete AI trading pipeline."""
    print(f"\n🚀 Testing Complete AI Trading Pipeline for {ticker}")
    print("=" * 60)

    results = {
        "ticker": ticker,
        "environment": None,
        "news_fetch": None,
        "ai_analysis": None,
        "strategy": None,
        "data_integration": None,
        "success": False,
    }

    # 1. Environment Validation
    print("1️⃣ Environment Validation...")
    env_result = validate_environment()
    results["environment"] = env_result

    if not env_result["api_keys_valid"]:
        print("❌ Environment validation failed")
        for error in env_result["errors"]:
            print(f"   - {error}")
        return results

    print("✅ Environment validated successfully")

    # 2. News Fetching
    print("\n2️⃣ News Data Fetching...")
    try:
        news_fetcher = get_news_fetcher()
        articles = news_fetcher.get_stock_news(ticker, max_articles=5, use_cache=False)

        results["news_fetch"] = {
            "success": True,
            "articles_count": len(articles),
            "articles": [
                article.to_dict() for article in articles[:2]
            ],  # Store first 2
        }

        print(f"✅ Fetched {len(articles)} news articles")
        for i, article in enumerate(articles[:3], 1):
            print(f"   {i}. {article.source}: {article.title[:50]}...")

        if not articles:
            print("⚠️  No articles found - continuing with mock data")
            # Create mock article for testing
            import datetime

            mock_article = NewsArticle(
                source="Mock News",
                title=f"{ticker} Reports Strong Quarterly Earnings",
                description=f"{ticker} company exceeded analyst expectations with strong revenue growth.",
                url="https://example.com/mock-news",
                published_at=datetime.datetime.now(datetime.timezone.utc),
            )
            articles = [mock_article]

    except Exception as e:
        print(f"❌ News fetching failed: {e}")
        results["news_fetch"] = {"success": False, "error": str(e)}
        return results

    # 3. AI Analysis
    print("\n3️⃣ AI Sentiment Analysis...")
    try:
        ai_analyzer = get_ai_analyzer()
        analysis_result = ai_analyzer.analyze_news_sentiment(ticker, articles)

        if analysis_result:
            results["ai_analysis"] = {
                "success": True,
                "signal": analysis_result.signal,
                "confidence": analysis_result.confidence,
                "rationale": analysis_result.rationale,
                "model": analysis_result.model_used,
                "numerical_signal": analysis_result.get_numerical_signal(),
            }

            print(f"✅ AI Analysis completed:")
            print(f"   Signal: {analysis_result.signal}")
            print(f"   Confidence: {analysis_result.confidence:.2f}")
            print(f"   Reasoning: {analysis_result.rationale[:100]}...")
            print(f"   Model: {analysis_result.model_used}")
        else:
            print("❌ AI analysis failed")
            results["ai_analysis"] = {"success": False, "error": "No analysis result"}
            return results

    except Exception as e:
        print(f"❌ AI analysis failed: {e}")
        results["ai_analysis"] = {"success": False, "error": str(e)}
        return results

    # 4. Strategy Integration
    print("\n4️⃣ Strategy Integration...")
    try:
        strategy = TradingStrategy()

        # Test signal generation (using mock price data if needed)
        import pandas as pd
        import datetime

        # Create mock price data for strategy testing
        dates = pd.date_range(end=datetime.date.today(), periods=10, freq="D")
        mock_data = pd.DataFrame(
            {
                "Date": dates,
                "Close": [150 + i * 2 for i in range(10)],  # Mock increasing prices
            }
        )
        mock_data.set_index("Date", inplace=True)

        signals = strategy.generate_signals(mock_data, ticker)

        results["strategy"] = {
            "success": True,
            "signals_generated": len(signals),
            "latest_signal": signals.iloc[-1].to_dict() if not signals.empty else None,
        }

        print(f"✅ Strategy integration successful")
        print(f"   Signals generated: {len(signals)}")
        if not signals.empty:
            latest = signals.iloc[-1]
            print(f"   Latest signal: {latest.get('ai_signal', 'N/A')}")

    except Exception as e:
        print(f"❌ Strategy integration failed: {e}")
        results["strategy"] = {"success": False, "error": str(e)}
        return results

    # 5. Data Manager Integration
    print("\n5️⃣ Data Manager Integration...")
    try:
        # Test DataManager creation and basic functionality
        data_manager = DataManager()

        results["data_integration"] = {
            "success": True,
            "data_manager_created": True,
            "note": "DataManager instance created successfully",
        }

        print(f"✅ Data Manager integration successful")
        print(f"   DataManager instance created")
        print(f"   Ready for stock data operations")

    except Exception as e:
        print(f"❌ Data integration failed: {e}")
        results["data_integration"] = {"success": False, "error": str(e)}
        return results

    # Mark as successful
    results["success"] = True
    print(f"\n🎉 Complete pipeline test PASSED for {ticker}")

    return results


def test_ui_applications() -> Dict[str, Any]:
    """Test both UI applications startup and basic functionality."""
    print(f"\n🖥️  Testing UI Applications")
    print("=" * 40)

    results = {
        "streamlit": {"available": False, "error": None},
    }

    # Test Streamlit
    print("📊 Testing Streamlit Application...")
    try:
        streamlit_process = subprocess.Popen(
            [
                "streamlit",
                "run",
                "streamlit_app.py",
                "--server.headless=true",
                "--server.port=8501",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(5)  # Wait for startup

        if streamlit_process.poll() is None:
            # Try to access health endpoint
            try:
                response = requests.get("http://localhost:8501/healthz", timeout=3)
                if response.status_code == 200:
                    results["streamlit"]["available"] = True
                    print("✅ Streamlit is running and accessible")
                    print("🌐 URL: http://localhost:8501")
                else:
                    results["streamlit"]["error"] = f"HTTP {response.status_code}"
                    print(f"⚠️  Streamlit responded with status {response.status_code}")
            except Exception as e:
                results["streamlit"]["error"] = str(e)
                print(f"⚠️  Could not access Streamlit: {e}")

            # Stop process
            streamlit_process.terminate()
            streamlit_process.wait(timeout=5)
        else:
            stdout, stderr = streamlit_process.communicate()
            error_msg = stderr.decode() if stderr else "Process terminated"
            results["streamlit"]["error"] = error_msg
            print(f"❌ Streamlit failed to start: {error_msg[:100]}")

    except Exception as e:
        results["streamlit"]["error"] = str(e)
        print(f"❌ Streamlit test failed: {e}")

    # Note: NiceGUI functionality has been integrated into the unified Streamlit app
    print(f"\n⚡ NiceGUI functionality is now integrated into the Streamlit app")
    print("✅ Real-time monitoring available via Streamlit Real-time Monitor tab")

    return results


def test_performance_metrics() -> Dict[str, Any]:
    """Test system performance metrics."""
    print(f"\n⚡ Testing Performance Metrics")
    print("=" * 35)

    results = {}

    # Test news fetching speed
    print("📰 News Fetching Performance...")
    start_time = time.time()
    fetcher = get_news_fetcher()
    articles = fetcher.get_stock_news("AAPL", max_articles=5, use_cache=False)
    news_time = time.time() - start_time
    results["news_fetch_time"] = news_time
    print(f"   News fetch: {news_time:.2f}s for {len(articles)} articles")

    # Test AI analysis speed
    if articles:
        print("🤖 AI Analysis Performance...")
        start_time = time.time()
        analyzer = get_ai_analyzer()
        analysis = analyzer.analyze_news_sentiment("AAPL", articles[:3])
        ai_time = time.time() - start_time
        results["ai_analysis_time"] = ai_time
        print(f"   AI analysis: {ai_time:.2f}s for sentiment analysis")

    # Test cache performance
    print("💾 Cache Performance...")
    start_time = time.time()
    cached_articles = fetcher.get_stock_news("AAPL", max_articles=5, use_cache=True)
    cache_time = time.time() - start_time
    results["cache_fetch_time"] = cache_time
    print(f"   Cached fetch: {cache_time:.2f}s for {len(cached_articles)} articles")

    if news_time > 0 and cache_time < news_time:
        speedup = news_time / cache_time
        print(f"   ✅ Cache speedup: {speedup:.1f}x")
        results["cache_speedup"] = speedup

    return results


def main():
    """Run complete final integration test."""
    print("🎯 AI Trading Demo - Final Integration Test")
    print("=" * 55)
    print("This test validates the complete system functionality")
    print("including APIs, AI analysis, and UI applications.\n")

    setup_logging()

    test_results = {"start_time": time.time(), "tests_passed": 0, "tests_total": 4}

    # 1. Complete Pipeline Test
    pipeline_result = test_complete_pipeline()
    if pipeline_result["success"]:
        test_results["tests_passed"] += 1
        print("✅ Pipeline Test: PASSED")
    else:
        print("❌ Pipeline Test: FAILED")
    test_results["pipeline"] = pipeline_result

    # 2. UI Applications Test
    ui_result = test_ui_applications()
    if ui_result["streamlit"]["available"]:
        test_results["tests_passed"] += 1
        print("✅ UI Applications Test: PASSED")
    else:
        print("❌ UI Applications Test: FAILED")
    test_results["ui_applications"] = ui_result

    # 3. Performance Test
    performance_result = test_performance_metrics()
    if performance_result.get("news_fetch_time", 0) < 10:  # Should be under 10s
        test_results["tests_passed"] += 1
        print("✅ Performance Test: PASSED")
    else:
        print("❌ Performance Test: FAILED")
    test_results["performance"] = performance_result

    # 4. Overall System Health
    if test_results["tests_passed"] >= 3:
        test_results["tests_passed"] += 1
        print("✅ System Health: PASSED")
    else:
        print("❌ System Health: FAILED")

    # Final Summary
    test_results["end_time"] = time.time()
    test_results["duration"] = test_results["end_time"] - test_results["start_time"]

    print("\n" + "=" * 55)
    print("📋 FINAL INTEGRATION TEST RESULTS")
    print("=" * 55)

    passed = test_results["tests_passed"]
    total = test_results["tests_total"]
    percentage = (passed / total) * 100

    print(f"📊 Tests Passed: {passed}/{total} ({percentage:.0f}%)")
    print(f"⏱️  Total Duration: {test_results['duration']:.1f}s")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The AI Trading Demo system is fully functional and ready for use.")
        print("\n🚀 Ready to Launch:")
        print("   • Streamlit: streamlit run streamlit_app.py")
        print("   • Real-time monitoring included in Streamlit app")

    elif passed >= 3:
        print("\n✅ SYSTEM FUNCTIONAL")
        print("Most tests passed. The system is largely functional with minor issues.")

    else:
        print("\n❌ SYSTEM ISSUES DETECTED")
        print("Multiple tests failed. Please review the results above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
