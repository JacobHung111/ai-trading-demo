# Algorithmic Trading Demo Project

Welcome to the Algorithmic Trading Demo Project! This application showcases a complete trading strategy analysis tool, combining historical data backtesting with a real-time monitoring dashboard. It features a hybrid architecture, utilizing both **Streamlit** for historical analysis and **NiceGUI** for live monitoring.

Notably, this project was developed with the assistance of an AI, demonstrating how modern AI tools can accelerate the prototyping and development of complex financial applications.

## Key Features

- **Hybrid Frontend Architecture**:
  - **Streamlit**: Used for historical data backtesting and strategy analysis, providing interactive charts and detailed reports.
  - **NiceGUI**: Powers a real-time monitoring dashboard with low-latency price updates and live signal notifications.
- **Modular Core Logic**: The core trading logic (data management, indicator calculation, strategy) is encapsulated in the `shared` directory, ensuring consistency and maintainability across both frontend applications.
- **SMA Crossover Strategy**: Implements the classic Simple Moving Average (SMA) crossover strategy:
  - **Buy Signal**: Generated when the short-term SMA (20-day) crosses above the long-term SMA (50-day).
  - **Sell Signal**: Generated when the short-term SMA (20-day) crosses below the long-term SMA (50-day).
- **Comprehensive Test Suite**: The project includes a full suite of unit tests in the `tests` directory, using the `pytest` framework to ensure the reliability and correctness of the core logic.
- **Clean Project Structure**: The project is organized logically, making it easy to understand, maintain, and extend.

## Project Structure

```
/
├───.gitignore
├───claude.md
├───nicegui_app.py         # NiceGUI real-time monitoring app
├───README.md
├───requirements-dev.txt   # Development dependencies
├───requirements.txt       # Production dependencies
├───streamlit_app.py       # Streamlit historical analysis app
├───shared/                # Shared core logic
│   ├───__init__.py
│   ├───config.py          # Centralized configuration
│   ├───data_manager.py    # Data fetching and caching
│   ├───indicators.py      # Technical indicator calculations
│   └───strategy.py        # Trading strategy implementation
└───tests/                 # Test suite
    ├───conftest.py
    ├───fixtures/
    │   └───sample_ohlcv.csv
    └───shared/
        ├───test_config.py
        ├───test_data_manager.py
        ├───test_indicators.py
        └───test_strategy.py
```

## Tech Stack

- **UI Frameworks**: Streamlit, NiceGUI
- **Data Processing**: Pandas
- **Financial Data Source**: yfinance
- **Data Visualization**: Plotly
- **Testing Framework**: Pytest

## Installation and Setup

1.  **Clone the repository**:

    ```bash
    git clone -b algorithmic-trading https://github.com/JacobHung111/ai-trading-demo.git algorithmic-trading-demo
    cd algorithmic-trading-demo
    ```

2.  **Create and activate a virtual environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    For development, install the development dependencies as well:

    ```bash
    pip install -r requirements-dev.txt
    ```

4.  **Run the applications**:

    - **To run the Streamlit app for historical analysis**:

      ```bash
      streamlit run streamlit_app.py
      ```

      The app will be available at `http://localhost:8501`.

    - **To run the NiceGUI app for real-time monitoring**:
      ```bash
      python nicegui_app.py
      ```
      The app will be available at `http://localhost:8081`.

## How to Use

### Streamlit Historical Analysis App

1.  Navigate to `http://localhost:8501`.
2.  Enter a stock ticker symbol (e.g., `AAPL`, `TSLA`) in the sidebar.
3.  Select the desired date range for analysis.
4.  Click the "Run Analysis" button.
5.  The application will display:
    - An interactive price chart with SMA indicators and trade signals.
    - A detailed table of all generated trade signals.
    - A statistical summary of the strategy's performance.

### NiceGUI Real-time Monitoring App

1.  Navigate to `http://localhost:8081`.
2.  Enter a ticker and click "Load Historical" to populate the chart with baseline data.
3.  Click "Start Monitoring" to begin receiving live updates.
4.  The dashboard will display:
    - The current stock price, updated in real-time.
    - The latest trading signal.
    - A dynamically updating price chart.
5.  You can adjust strategy parameters (like SMA periods) and the update interval via the "Settings" menu.

## Running Tests

To ensure all components are working correctly, run the test suite:

```bash
pytest
```

## Future Goals

- **Integrate AI/ML Models**: Evolve the project from "AI-assisted" to truly "AI-powered" by incorporating machine learning models (e.g., LSTM, Transformers) to predict price movements or generate more sophisticated trading signals.
- **Add More Indicators**: Implement other popular technical indicators like RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence).
- **Advanced Backtesting Engine**: Enhance the backtester to account for transaction costs, slippage, and other real-world factors for more accurate performance evaluation.
- **Portfolio Management**: Add features for managing a virtual portfolio and simulating trades.

---

**Disclaimer**: This project is for educational and demonstration purposes only and does not constitute financial advice. All trading decisions are made at your own risk.
