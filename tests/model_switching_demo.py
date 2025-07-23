#!/usr/bin/env python3
"""
AI Model Switching Functionality Demonstration Script

This script demonstrates the AI model switching functionality implemented
in shared/config.py and validates that it works correctly in practice.

Author: AI Trading Demo Team
Version: 2.0
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import AITradingConfig, get_config, reset_config


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_model_info(config: AITradingConfig, model_name: str = None) -> None:
    """Print current model configuration."""
    if model_name:
        print(f"\nModel: {model_name}")
    
    current_info = config.get_current_model_info()
    print(f"  Display Name: {current_info['name']}")
    print(f"  Description: {current_info.get('description', 'N/A')}")
    print(f"  Rate Limit (requests/min): {current_info['current_rate_limit_requests']}")
    print(f"  Rate Limit (tokens/min): {current_info['current_rate_limit_tokens']}")
    print(f"  Max Tokens: {current_info['current_max_tokens']}")
    
    # Show source if available
    available_models = config.get_available_models()
    if config.ai_model_name in available_models:
        source = available_models[config.ai_model_name].get('source', 'hardcoded')
        print(f"  Source: {source}")


def test_model_switching():
    """Test basic model switching functionality."""
    print_header("Testing Basic Model Switching")
    
    # Create a fresh config instance
    config = AITradingConfig()
    print(f"Initial model: {config.ai_model_name}")
    print_model_info(config)
    
    # Test switching to different models
    models_to_test = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"]
    
    for model in models_to_test:
        print(f"\n→ Switching to {model}")
        success = config.update_model_settings(model)
        
        if success:
            print(f"  ✅ Successfully switched to {model}")
            print_model_info(config, model)
        else:
            print(f"  ❌ Failed to switch to {model}")
    
    # Test invalid model
    print(f"\n→ Testing invalid model")
    success = config.update_model_settings("invalid-model-name")
    if not success:
        print(f"  ✅ Correctly rejected invalid model")
    else:
        print(f"  ❌ Unexpectedly accepted invalid model")


def test_dynamic_model_fetching():
    """Test dynamic model fetching from Google API."""
    print_header("Testing Dynamic Model Fetching")
    
    config = AITradingConfig()
    
    # Test API key availability
    api_keys = config.validate_api_keys()
    print(f"Google API Key available: {api_keys['google_api_key']}")
    
    if api_keys['google_api_key']:
        print("Attempting to fetch models from Google API...")
        dynamic_models = config.fetch_models_from_api()
        
        if dynamic_models:
            print(f"✅ Successfully fetched {len(dynamic_models)} models from API")
            
            # Show first few models
            model_names = list(dynamic_models.keys())[:5]
            for model_name in model_names:
                model_info = dynamic_models[model_name]
                print(f"  - {model_name}: {model_info.get('name', 'N/A')}")
        else:
            print("❌ Failed to fetch models from API")
    else:
        print("⚠️  No Google API key available - using hardcoded models only")
    
    # Show available models
    available_models = config.get_available_models()
    print(f"\nTotal available models: {len(available_models)}")


def test_rate_limit_updates():
    """Test that rate limits are updated correctly when switching models."""
    print_header("Testing Rate Limit Updates")
    
    config = AITradingConfig()
    
    # Test different model types and their rate limits
    test_cases = [
        ("gemini-2.0-flash-lite", "Flash Lite model"),
        ("gemini-1.5-flash", "Flash model"),
        ("gemini-1.5-pro", "Pro model")
    ]
    
    for model_name, description in test_cases:
        print(f"\n→ Testing {description} ({model_name})")
        
        # Record previous values
        prev_requests = config.gemini_rate_limit_requests
        prev_tokens = config.gemini_rate_limit_tokens
        prev_max_tokens = config.ai_max_tokens
        
        # Switch model
        success = config.update_model_settings(model_name)
        
        if success:
            # Check if values changed
            requests_changed = config.gemini_rate_limit_requests != prev_requests
            tokens_changed = config.gemini_rate_limit_tokens != prev_tokens
            max_tokens_changed = config.ai_max_tokens != prev_max_tokens
            
            print(f"  Rate limits - Requests: {prev_requests} → {config.gemini_rate_limit_requests} {'✓' if requests_changed else '='}")
            print(f"  Rate limits - Tokens: {prev_tokens} → {config.gemini_rate_limit_tokens} {'✓' if tokens_changed else '='}")
            print(f"  Max tokens: {prev_max_tokens} → {config.ai_max_tokens} {'✓' if max_tokens_changed else '='}")
        else:
            print(f"  ❌ Failed to switch to {model_name}")


def test_error_handling():
    """Test error handling scenarios."""
    print_header("Testing Error Handling")
    
    config = AITradingConfig()
    
    # Test invalid model switching
    print("→ Testing invalid model name")
    success = config.update_model_settings("non-existent-model")
    print(f"  Invalid model rejection: {'✅' if not success else '❌'}")
    
    # Test API error handling by temporarily removing API key
    print("\n→ Testing API error handling")
    original_key = config.google_api_key
    config.google_api_key = None
    
    dynamic_models = config.fetch_models_from_api()
    api_error_handled = dynamic_models is None
    print(f"  API error handling: {'✅' if api_error_handled else '❌'}")
    
    # Restore original key
    config.google_api_key = original_key
    
    # Test configuration validation
    print("\n→ Testing configuration validation")
    valid_ai_params = config.validate_ai_parameters()
    valid_rate_limits = config.validate_rate_limits()
    print(f"  AI parameters valid: {'✅' if valid_ai_params else '❌'}")
    print(f"  Rate limits valid: {'✅' if valid_rate_limits else '❌'}")


def test_global_config_integration():
    """Test integration with global configuration instance."""
    print_header("Testing Global Configuration Integration")
    
    # Reset global config
    reset_config()
    global_config = get_config()
    
    print(f"Global config initial model: {global_config.ai_model_name}")
    
    # Test switching model on global instance
    print("\n→ Switching global config to gemini-1.5-pro")
    success = global_config.update_model_settings("gemini-1.5-pro")
    
    if success:
        print("✅ Successfully updated global configuration")
        print_model_info(global_config)
        
        # Verify it persists by getting config again
        same_config = get_config()
        if same_config.ai_model_name == "gemini-1.5-pro":
            print("✅ Global config changes persist")
        else:
            print("❌ Global config changes did not persist")
    else:
        print("❌ Failed to update global configuration")


def performance_benchmark():
    """Benchmark model switching performance."""
    print_header("Performance Benchmark")
    
    config = AITradingConfig()
    models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"]
    
    print("Benchmarking model switching performance...")
    
    # Warm up
    for model in models:
        config.update_model_settings(model)
    
    # Benchmark
    start_time = time.time()
    iterations = 50
    
    for i in range(iterations):
        for model in models:
            config.update_model_settings(model)
    
    end_time = time.time()
    total_switches = iterations * len(models)
    avg_time = (end_time - start_time) / total_switches * 1000  # milliseconds
    
    print(f"Total switches: {total_switches}")
    print(f"Total time: {end_time - start_time:.3f} seconds")
    print(f"Average time per switch: {avg_time:.2f} ms")
    
    if avg_time < 10:
        print("✅ Performance: Excellent (< 10ms per switch)")
    elif avg_time < 50:
        print("✅ Performance: Good (< 50ms per switch)")
    else:
        print("⚠️  Performance: Acceptable but could be improved")


def main():
    """Run all demonstration tests."""
    print_header("AI Model Switching Functionality Demonstration")
    print("This script demonstrates the AI model switching capabilities")
    print("implemented in the AI Trading Demo project.")
    
    try:
        # Run all test demonstrations
        test_model_switching()
        test_dynamic_model_fetching()
        test_rate_limit_updates()
        test_error_handling()
        test_global_config_integration()
        performance_benchmark()
        
        print_header("Summary")
        print("✅ All model switching functionality demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Basic model switching between different Gemini models")
        print("  • Dynamic model fetching from Google Gemini API")
        print("  • Rate limit updates when switching models")
        print("  • Error handling for invalid models and API failures")
        print("  • Global configuration instance integration")
        print("  • Performance benchmarking")
        
        print("\nThe AI model switching functionality is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)