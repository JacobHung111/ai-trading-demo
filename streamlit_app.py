"""
AI Trading Demo - Main Streamlit Application

This is the main entry point for the AI Trading Demo application. It provides
a user-friendly interface for analyzing stock data using Simple Moving Average
crossover strategy, displaying interactive charts and trading signals.

Author: AI Trading Demo Team
Version: 1.0
"""

import datetime
from typing import Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our shared components and configuration
from shared.config import get_config
from shared.data_manager import load_data_with_streamlit_cache, validate_date_inputs_ui
from shared.indicators import calculate_indicators_with_ui_feedback
from shared.strategy import generate_trading_signals_with_ui_feedback, get_signal_summary_with_ui_feedback


def main() -> None:
    """Main application function that orchestrates the Streamlit UI and trading analysis.
    
    This function sets up the Streamlit page configuration, handles user input,
    processes the trading data, and displays the results through interactive
    visualizations and statistical summaries.
    """
    # Page configuration
    st.set_page_config(
        page_title="AI Trading Demo",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Application header
    st.title("📈 AI Trading Demo")
    st.markdown("""
    **Welcome to the AI Trading Strategy Demonstration**
    
    This application demonstrates a Simple Moving Average (SMA) crossover trading strategy.
    The strategy generates buy signals when the 20-day SMA crosses above the 50-day SMA,
    and sell signals when the 20-day SMA crosses below the 50-day SMA.
    
    > **Disclaimer**: This is for educational purposes only. Not financial advice.
    """)
    
    # Sidebar for user inputs
    st.sidebar.header("📊 Trading Parameters")
    
    # Load configuration
    config = get_config()
    
    # Stock ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker Symbol",
        value=config.default_ticker,
        help="Enter a valid stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
    ).upper()
    
    # Date range selection
    st.sidebar.subheader("Date Range Selection")
    
    # Default date range (1 year back from today)
    default_end_date = datetime.date.today()
    default_start_date = default_end_date - datetime.timedelta(days=365)
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start_date,
        max_value=default_end_date,
        help="Select the start date for historical data analysis"
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end_date,
        min_value=start_date,
        max_value=default_end_date,
        help="Select the end date for historical data analysis"
    )
    
    # Validate date inputs
    if not validate_date_inputs_ui(start_date, end_date):
        st.stop()
    
    # Analysis button
    analyze_button = st.sidebar.button(
        "🚀 Run Analysis",
        type="primary",
        help="Click to fetch data and generate trading signals"
    )
    
    # Main content area
    if analyze_button or st.session_state.get('data_loaded', False):
        
        with st.spinner(f"Loading data for {ticker}..."):
            # Load stock data
            stock_data = load_data_with_streamlit_cache(ticker, start_date, end_date)
            
            # Check if data was successfully loaded
            if stock_data.empty:
                st.error("❌ Failed to load data. Please check your inputs and try again.")
                st.stop()
            
            # Calculate indicators
            data_with_indicators = calculate_indicators_with_ui_feedback(stock_data)
            
            if data_with_indicators.empty:
                st.error("❌ Insufficient data for indicator calculation. Please try a longer date range.")
                st.stop()
            
            # Generate trading signals
            final_data = generate_trading_signals_with_ui_feedback(data_with_indicators)
            
            # Store data in session state
            st.session_state['data_loaded'] = True
            st.session_state['final_data'] = final_data
            st.session_state['ticker'] = ticker
    
    # Display results if data is available
    if st.session_state.get('data_loaded', False):
        display_results(
            st.session_state['final_data'],
            st.session_state['ticker']
        )


def display_results(data: pd.DataFrame, ticker: str) -> None:
    """Displays the trading analysis results including charts and statistics.
    
    This function creates the main results display area with interactive charts,
    trading signals, and statistical summaries of the analysis.
    
    Args:
        data (pd.DataFrame): The processed data containing OHLCV data, indicators, and signals.
        ticker (str): The stock ticker symbol for display purposes.
    """
    # Results header
    st.header(f"📊 Analysis Results for {ticker}")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Price Chart", "📋 Trading Signals", "📊 Statistics"])
    
    with tab1:
        st.subheader("Stock Price with Moving Averages")
        display_price_chart(data, ticker)
    
    with tab2:
        st.subheader("Trading Signals Overview")
        display_signals_table(data)
    
    with tab3:
        st.subheader("Trading Statistics")
        display_statistics(data)


def display_price_chart(data: pd.DataFrame, ticker: str) -> None:
    """Creates and displays an interactive price chart with moving averages and signals.
    
    This function generates a comprehensive chart showing stock price, moving averages,
    and trading signals using Plotly for interactivity.
    
    Args:
        data (pd.DataFrame): The processed data containing OHLCV data, indicators, and signals.
        ticker (str): The stock ticker symbol for chart title.
    """
    # Create subplots for price and volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{ticker} Stock Price with Trading Signals', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Add stock price line
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>',
            text=[ticker] * len(data)
        ),
        row=1, col=1
    )
    
    # Add SMA20 line
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['SMA20'],
            mode='lines',
            name='SMA20',
            line=dict(color='#ff7f0e', width=1.5),
            hovertemplate='<b>SMA20</b><br>Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add SMA50 line
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['SMA50'],
            mode='lines',
            name='SMA50',
            line=dict(color='#2ca02c', width=1.5),
            hovertemplate='<b>SMA50</b><br>Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add buy signals
    buy_signals = data[data['Signal'] == 1]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['Date'],
                y=buy_signals['Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(color='darkgreen', width=2)
                ),
                hovertemplate='<b>BUY SIGNAL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add sell signals
    sell_signals = data[data['Signal'] == -1]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['Date'],
                y=sell_signals['Close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(color='darkred', width=2)
                ),
                hovertemplate='<b>SELL SIGNAL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name='Volume',
            marker_color='rgba(158, 158, 158, 0.5)',
            hovertemplate='<b>Volume</b><br>Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        xaxis2_title="Date",
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


def display_signals_table(data: pd.DataFrame) -> None:
    """Displays a table of trading signals with relevant details.
    
    This function creates a filtered view of the data showing only dates
    where trading signals were generated, along with relevant price information.
    
    Args:
        data (pd.DataFrame): The processed data containing trading signals.
    """
    # Filter data to show only signals
    signals_data = data[data['Signal'] != 0].copy()
    
    if signals_data.empty:
        st.info("ℹ️ No trading signals generated for the selected period.")
        return
    
    # Prepare display data
    signals_data['Signal_Type'] = signals_data['Signal'].map({
        1: '🟢 BUY',
        -1: '🔴 SELL'
    })
    
    # Select and rename columns for display
    display_columns = {
        'Date': 'Date',
        'Close': 'Price ($)',
        'SMA20': 'SMA20 ($)',
        'SMA50': 'SMA50 ($)',
        'Signal_Type': 'Signal',
        'Volume': 'Volume'
    }
    
    signals_display = signals_data[list(display_columns.keys())].rename(columns=display_columns)
    
    # Format numeric columns
    signals_display['Price ($)'] = signals_display['Price ($)'].round(2)
    signals_display['SMA20 ($)'] = signals_display['SMA20 ($)'].round(2)
    signals_display['SMA50 ($)'] = signals_display['SMA50 ($)'].round(2)
    signals_display['Volume'] = signals_display['Volume'].apply(lambda x: f"{x:,.0f}")
    
    # Display the table
    st.dataframe(
        signals_display,
        use_container_width=True,
        hide_index=True
    )
    
    # Show summary
    signal_summary = get_signal_summary_with_ui_feedback(data)
    if signal_summary:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Signals", signal_summary['total_signals'])
        
        with col2:
            st.metric("Buy Signals", signal_summary['buy_signals'])
        
        with col3:
            st.metric("Sell Signals", signal_summary['sell_signals'])


def display_statistics(data: pd.DataFrame) -> None:
    """Displays statistical analysis and performance metrics.
    
    This function provides various statistical insights about the trading
    strategy performance and data characteristics.
    
    Args:
        data (pd.DataFrame): The processed data for statistical analysis.
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Overview")
        
        # Basic statistics
        st.write("**Time Period:**")
        st.write(f"• Start Date: {data['Date'].min().strftime('%Y-%m-%d')}")
        st.write(f"• End Date: {data['Date'].max().strftime('%Y-%m-%d')}")
        st.write(f"• Total Days: {len(data)}")
        
        st.write("**Price Statistics:**")
        st.write(f"• Highest Price: ${data['Close'].max():.2f}")
        st.write(f"• Lowest Price: ${data['Close'].min():.2f}")
        st.write(f"• Average Price: ${data['Close'].mean():.2f}")
        st.write(f"• Price Volatility (Std): ${data['Close'].std():.2f}")
    
    with col2:
        st.subheader("📈 Trading Performance")
        
        signal_summary = get_signal_summary_with_ui_feedback(data)
        
        if signal_summary:
            st.write("**Signal Analysis:**")
            st.write(f"• Total Trading Days: {signal_summary['total_days']}")
            st.write(f"• Signal Days: {signal_summary['total_signals']}")
            st.write(f"• Buy Signals: {signal_summary['buy_signals']}")
            st.write(f"• Sell Signals: {signal_summary['sell_signals']}")
            
            if signal_summary['total_days'] > 0:
                signal_rate = (signal_summary['total_signals'] / signal_summary['total_days']) * 100
                st.write(f"• Signal Rate: {signal_rate:.1f}%")
        else:
            st.info("No signals generated for statistical analysis.")
    
    # Additional insights
    st.subheader("🔍 Technical Analysis Insights")
    
    # Calculate current position of moving averages
    latest_data = data.iloc[-1]
    sma_position = "bullish" if latest_data['SMA20'] > latest_data['SMA50'] else "bearish"
    
    st.write(f"**Current Market Position:** {sma_position.title()}")
    st.write(f"• Current SMA20: ${latest_data['SMA20']:.2f}")
    st.write(f"• Current SMA50: ${latest_data['SMA50']:.2f}")
    
    # Show recent signal if any
    recent_signals = data[data['Signal'] != 0].tail(1)
    if not recent_signals.empty:
        last_signal = recent_signals.iloc[-1]
        signal_type = "BUY" if last_signal['Signal'] == 1 else "SELL"
        st.write(f"**Most Recent Signal:** {signal_type} on {last_signal['Date'].strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()