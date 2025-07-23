# AI Trading Demo - Intelligent Stock Analysis Platform

![AI Trading Demo](https://img.shields.io/badge/AI-Trading%20Demo-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit)
![Google Gemini](https://img.shields.io/badge/Google-Gemini%20AI-yellow?style=for-the-badge&logo=google)

A sophisticated AI-powered trading analysis platform that combines news sentiment analysis with financial data visualization to provide intelligent trading insights.

## Overview

This application demonstrates advanced AI integration in financial analysis by using **Google Gemini AI** to analyze news sentiment and generate trading signals. The platform provides:

- **AI-Powered News Analysis** - Real-time sentiment analysis of financial news
- **Interactive Data Visualization** - Comprehensive charts and trading signal visualization  
- **Intelligent Signal Generation** - BUY/SELL/HOLD recommendations based on AI analysis
- **Historical Performance Tracking** - Track AI decision accuracy over time
- **Real-Time Monitoring** - Live price updates and continuous analysis

## Key Features

### AI Integration
- **Google Gemini API** integration for sophisticated news sentiment analysis
- Advanced prompt engineering for financial context understanding
- Confidence scoring for each trading recommendation
- Intelligent caching to optimize API usage

### Data & Analytics
- **Yahoo Finance** integration for real-time stock data
- **NewsAPI.org** for comprehensive financial news coverage
- Interactive **Plotly** charts with AI signal overlays
- Historical performance metrics and trend analysis

### Technical Excellence
- Modular architecture with clean separation of concerns
- Comprehensive error handling and graceful degradation
- Rate limiting and quota management
- Extensive test coverage with mocking for external APIs

## Architecture

```
ai-trading-demo/
├── streamlit_app.py          # Main Streamlit application
├── core/                     # Core business logic
│   ├── config.py            # Configuration management
│   ├── data_manager.py      # Stock data fetching
│   ├── news_fetcher.py      # News API integration
│   ├── ai_analyzer.py       # Gemini AI integration
│   └── strategy.py          # Trading strategy logic
├── ui/                      # User interface components
│   └── components.py        # Streamlit UI components
├── utils/                   # Utility modules
│   ├── api.py              # API utilities
│   ├── charts.py           # Chart creation
│   └── errors.py            # Error handling
└── tests/                   # Comprehensive test suite
```

## Quick Start

### Prerequisites

- **Python 3.9+** installed on your system
- **Google Gemini API Key** ([Get yours here](https://makersuite.google.com/app/apikey))
- **NewsAPI.org Key** ([Register here](https://newsapi.org/register))

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai-trading-demo
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create a .env file or export environment variables
export GOOGLE_API_KEY="your_gemini_api_key_here"
export NEWS_API_KEY="your_newsapi_key_here"
```

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

5. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Start analyzing stocks with AI-powered insights!

## How It Works

### 1. Data Collection
- Fetches real-time stock price data from Yahoo Finance
- Retrieves relevant financial news from NewsAPI.org
- Caches data efficiently to minimize API calls

### 2. AI Analysis
- Sends news headlines to Google Gemini AI with carefully crafted prompts
- Analyzes sentiment and potential market impact
- Generates structured responses with confidence scores

### 3. Signal Generation
- **BUY (1)**: Positive sentiment detected in news analysis
- **SELL (-1)**: Negative sentiment detected in news analysis
- **HOLD (0)**: Neutral sentiment or insufficient data

### 4. Visualization
- Interactive charts showing price movements and AI signals
- Detailed analysis tables with reasoning for each recommendation
- Performance metrics and historical accuracy tracking

## Use Cases

- **Educational**: Learn about AI applications in finance
- **Research**: Analyze correlation between news sentiment and stock movements
- **Demonstration**: Showcase AI integration in financial applications
- **Analysis**: Historical backtesting of AI-driven trading strategies

## Error Handling & Robustness

- **API Quota Management**: Automatic demo mode when quotas are exhausted
- **Network Resilience**: Retry logic with exponential backoff
- **Graceful Degradation**: Fallback mechanisms for service unavailability
- **User-Friendly Messages**: Clear error explanations and suggested actions

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=core --cov=ui --cov=utils --cov-report=html

# Run specific test categories
pytest tests/shared/test_ai_analyzer.py -v
```

## Configuration

The application supports various configuration options through `core/config.py`:

- **AI Model Selection**: Switch between different Gemini models
- **Rate Limiting**: Customize API call frequency
- **Caching**: Adjust cache duration for different data types
- **Analysis Parameters**: Configure news article limits and analysis depth

## Important Disclaimers

- **Not Financial Advice**: This application is for educational and demonstration purposes only
- **Showcase Project**: Designed to demonstrate AI integration, not generate profitable trading strategies
- **No Real Trading**: Never connects to real brokerages or executes actual trades
- **News-Based Only**: Signals are based solely on news sentiment, not comprehensive market analysis



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Google Gemini AI** for advanced language model capabilities
- **Streamlit** for the excellent web application framework
- **Yahoo Finance** for reliable financial data
- **NewsAPI.org** for comprehensive news coverage
- **Plotly** for interactive data visualization

## Support

For questions, issues, or suggestions:

1. Check the [Issues](../../issues) section
2. Review the `CLAUDE.md` file for detailed architecture information
3. Run the test suite to verify your environment setup

---

**Built with love to demonstrate the power of AI in financial analysis**
