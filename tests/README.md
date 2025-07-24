# AI Trading Demo - Test Suite

Complete testing framework for the AI-powered trading demonstration system.

## 📁 Test Structure

### Unit Tests (`shared/`)
Individual component testing with mocks and fixtures.

- **`test_ai_analyzer.py`** - AI sentiment analysis and signal generation (42 tests)
- **`test_config_model_switching.py`** - AI model configuration and switching (25 tests)
- **`test_data_manager.py`** - Stock data fetching and caching (33 tests)
- **`test_news_fetcher.py`** - News API integration and article processing (26 tests)
- **`test_strategy.py`** - Trading strategy logic and signal computation (24 tests)

### Integration Tests (Root Level)
End-to-end system testing with real APIs.

- **`test_integration.py`** - Core pipeline integration with API connectivity
- **`test_system.py`** - Complete system validation and performance testing

### Test Data (`fixtures/`)
Reusable test data and mock responses.

- **`sample_ohlcv.csv`** - Mock stock price data for strategy testing

## 🧪 Running Tests

### Quick Test Commands
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=core --cov=ui --cov=utils --cov-report=html

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration             # Integration tests only
pytest -m "not slow"              # Skip slow tests
```

### Unit Tests (Mock Data)
```bash
# Run all unit tests
pytest tests/shared/ -v

# Specific component
pytest tests/shared/test_ai_analyzer.py -v
pytest tests/shared/test_strategy.py::TestAITradingStrategy -v
```

### Integration Tests (Real APIs)
**⚠️ Requires API keys in environment variables**

```bash
# Core system integration
pytest tests/test_integration.py -v

# Complete system validation  
pytest tests/test_system.py -v

# Skip tests requiring API keys
pytest -m "not ai_required and not news_required"
```

## 📊 Test Coverage

- **Unit Tests**: 90%+ coverage for `shared/` modules
- **Integration Tests**: 100% pipeline coverage with real APIs
- **UI Tests**: Both Streamlit and NiceGUI startup validation
- **Performance Tests**: Response time and caching validation

## 🔧 Test Configuration

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

### Dependencies
```bash
# Install test dependencies
pip install -r requirements-dev.txt
```

## 📝 Test Categories

| Test Type | Purpose | Data Source | Duration |
|-----------|---------|-------------|----------|
| **Unit** | Component logic | Mock data | ~30s |
| **Integration** | API connectivity | Real APIs | ~60s |
| **UI** | Application startup | Real APIs | ~45s |
| **System** | End-to-end validation | Real APIs | ~90s |

## ✅ Quality Gates

All tests must pass before deployment:
1. Unit tests achieve 90%+ coverage
2. Integration tests complete successfully  
3. UI applications start without errors
4. System performance meets thresholds (<10s total)