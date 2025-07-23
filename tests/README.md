# AI Trading Demo - Test Suite

Complete testing framework for the AI-powered trading demonstration system.

## 📁 Test Structure

### Unit Tests (`shared/`)
Individual component testing with mocks and fixtures.

- **`test_config.py`** - Configuration management and environment validation
- **`test_data_manager.py`** - Stock data fetching and caching
- **`test_news_fetcher.py`** - News API integration and article processing  
- **`test_ai_analyzer.py`** - AI sentiment analysis and signal generation
- **`test_strategy.py`** - Trading strategy logic and signal computation

### Integration Tests (Root Level)
End-to-end system testing with real APIs.

- **`integration_test.py`** - Core pipeline integration (News → AI → Strategy)
- **`api_compatibility_test.py`** - API response format validation and updates
- **`ui_integration_test.py`** - UI applications testing (Streamlit + NiceGUI)
- **`system_integration_test.py`** - Complete system validation and performance

### Test Data (`fixtures/`)
Reusable test data and mock responses.

- **`sample_ohlcv.csv`** - Mock stock price data for strategy testing

## 🧪 Running Tests

### Unit Tests (Mock Data)
```bash
# Run all unit tests
pytest tests/shared/ -v

# With coverage report
pytest tests/shared/ --cov=shared --cov-report=term-missing

# Specific component
pytest tests/shared/test_ai_analyzer.py -v
```

### Integration Tests (Real APIs)
**⚠️ Requires API keys in .env file**

```bash
# Core system integration
python tests/integration_test.py

# API compatibility validation  
python tests/api_compatibility_test.py

# UI applications testing
python tests/ui_integration_test.py

# Complete system validation
python tests/system_integration_test.py
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