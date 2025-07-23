"""
AI Trading Demo - Unified Trading Analysis & Monitoring Platform

This is the unified AI Trading Demo application that combines comprehensive historical
analysis with real-time monitoring capabilities. The application provides AI-powered
news sentiment analysis, interactive charts, detailed trading signals, and live
price monitoring in a single, integrated platform.

Features:
- Historical AI-powered trading signal analysis
- Real-time price monitoring and live updates
- Interactive charts with AI signal visualization
- Comprehensive trading statistics and performance metrics
- News sentiment analysis using Google Gemini AI

Author: AI Trading Demo Team
Version: 2.1 (Unified Platform Architecture)
"""

import datetime
import time
from typing import Dict
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our components and configuration
from core.config import get_config
from core.data_manager import (
    load_data_with_streamlit_cache,
    validate_date_inputs_ui,
    DataManager,
)
from core.strategy import (
    generate_trading_signals_with_ui_feedback,
    get_signal_summary_with_ui_feedback,
)


def main() -> None:
    """Main application function that orchestrates the Streamlit UI and trading analysis.

    This function sets up the Streamlit page configuration, handles user input,
    processes the trading data, and displays the results through interactive
    visualizations and statistical summaries.
    """
    # Page configuration
    st.set_page_config(
        page_title="AI Trading Platform - Analysis & Monitoring",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Application header
    st.title("🤖 AI Trading Platform - Analysis & Monitoring")
    st.markdown(
        """
    **Unified AI-Powered Trading Analysis & Real-time Monitoring Platform**
    
    This integrated platform combines comprehensive historical analysis with real-time monitoring capabilities. 
    Features include AI-powered news sentiment analysis, interactive visualizations, detailed trading signals, 
    and live price monitoring - all in one unified interface.
    
    ### 🚀 **Key Features**:
    - **📊 Historical Analysis**: Deep AI analysis of past trading signals and performance
    - **🤖 AI Trading Signals**: Google Gemini AI analyzes news sentiment for BUY/SELL/HOLD recommendations  
    - **⚡ Real-time Monitoring**: Live price updates and signal tracking
    - **📈 Interactive Charts**: Advanced visualizations with AI signal overlays
    
    📅 **Date Range**: Due to NewsAPI limitations, analysis covers the **most recent 30 days**.
    The AI analyzes each day individually - only days with relevant news generate signals.
    
    > **Disclaimer**: This is for educational and demonstration purposes only. Not financial advice.
    """
    )

    # Sidebar for user inputs
    st.sidebar.header("📊 Trading Parameters")

    # API Status Check
    display_api_status_check()

    # AI Model Selection
    display_ai_model_selection()

    # Load configuration
    config = get_config()

    # Stock ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker Symbol",
        value=config.default_ticker,
        help="Enter a valid stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
    ).upper()

    # Date range selection
    st.sidebar.subheader("Date Range Selection")

    # Default date range (30 days back from today - NewsAPI limitation)
    default_end_date = datetime.date.today()
    default_start_date = default_end_date - datetime.timedelta(
        days=29
    )  # 29 days to stay within 30-day limit

    # Calculate NewsAPI date limitation
    newsapi_limit_date = datetime.date.today() - datetime.timedelta(days=30)

    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start_date,
        min_value=newsapi_limit_date,  # Prevent selection beyond NewsAPI limit
        max_value=default_end_date,
        help=f"Select the start date (NewsAPI limit: earliest selectable date is {newsapi_limit_date})",
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end_date,
        min_value=start_date,
        max_value=default_end_date,
        help="Select the end date for historical data analysis",
    )

    # Validate date inputs
    if not validate_date_inputs_ui(start_date, end_date):
        st.stop()

    # Check NewsAPI date limitations (30 days for free plan)
    newsapi_limit_date = datetime.date.today() - datetime.timedelta(days=30)
    if start_date < newsapi_limit_date:
        st.sidebar.error(
            f"⚠️ NewsAPI Limitation: Free plan only supports news from the last 30 days. Please select a date after {newsapi_limit_date}."
        )
        st.stop()

    # Analysis button
    analyze_button = st.sidebar.button(
        "🚀 Run Analysis",
        type="primary",
        help="Click to fetch data and generate trading signals",
    )

    # Main content area
    if analyze_button:

        with st.spinner(f"Loading data for {ticker}..."):
            # Load stock data
            stock_data = load_data_with_streamlit_cache(ticker, start_date, end_date)

            # Check if data was successfully loaded
            if stock_data.empty:
                st.error(
                    "❌ Failed to load data. Please check your inputs and try again."
                )
                st.stop()

            # Generate AI-powered trading signals
            final_data = generate_trading_signals_with_ui_feedback(
                stock_data, ticker=ticker
            )

            # Store data in session state
            st.session_state["data_loaded"] = True
            st.session_state["final_data"] = final_data
            st.session_state["ticker"] = ticker

    # Display results if data is available
    if st.session_state.get("data_loaded", False):
        display_results(st.session_state["final_data"], st.session_state["ticker"])


def display_results(data: pd.DataFrame, ticker: str) -> None:
    """Displays the trading analysis results including charts and statistics.

    This function creates the main results display area with interactive charts,
    integrated AI analysis, and statistical summaries.

    Args:
        data (pd.DataFrame): The processed data containing OHLCV data, indicators, and signals.
        ticker (str): The stock ticker symbol for display purposes.
    """
    # Results header
    st.header(f"📊 Analysis Results for {ticker}")

    # Create integrated tabs including real-time monitoring
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Price Chart",
            "🤖 AI Trading Analysis",
            "⚡ Real-time Monitor",
            "📊 Statistics",
        ]
    )

    with tab1:
        st.subheader("Stock Price with AI Signals")
        display_price_chart(data, ticker)

    with tab2:
        # This is the integrated AI analysis tab that combines trading signals and AI insights
        from ui.components import display_integrated_ai_analysis

        display_integrated_ai_analysis(data, ticker)

    with tab3:
        # Real-time monitoring functionality from NiceGUI app
        display_realtime_monitoring(data, ticker)

    with tab4:
        st.subheader("Trading Statistics")
        display_statistics(data)


def display_price_chart(data: pd.DataFrame, ticker: str) -> None:
    """Creates and displays an interactive price chart with AI-generated trading signals.

    This function generates a comprehensive chart showing stock price and
    AI-powered trading signals using Plotly for interactivity.

    Args:
        data (pd.DataFrame): The processed data containing OHLCV data and AI signals.
        ticker (str): The stock ticker symbol for chart title.
    """
    # Create subplots for price and volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"{ticker} Stock Price with AI Signals", "Volume"),
        row_heights=[0.7, 0.3],
    )

    # Add stock price line
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
            text=[ticker] * len(data),
        ),
        row=1,
        col=1,
    )

    # Add AI confidence indicator as subtle background for signals
    if "AI_Confidence" in data.columns:
        signal_data = data[data["Signal"] != 0]
        if not signal_data.empty:
            for _, row in signal_data.iterrows():
                signal_color = (
                    "rgba(0,200,0,0.1)" if row["Signal"] == 1 else "rgba(200,0,0,0.1)"
                )
                fig.add_vrect(
                    x0=row["Date"] - pd.Timedelta(hours=12),
                    x1=row["Date"] + pd.Timedelta(hours=12),
                    fillcolor=signal_color,
                    opacity=max(0.1, row["AI_Confidence"] * 0.3),
                    line_width=0,
                    row=1,
                )

    # Add buy signals
    buy_signals = data[data["Signal"] == 1]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals["Date"],
                y=buy_signals["Close"],
                mode="markers",
                name="Buy Signal",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="green",
                    line=dict(color="darkgreen", width=2),
                ),
                hovertemplate="<b>AI BUY SIGNAL</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>AI Confidence: %{customdata[0]:.1%}<extra></extra>",
                customdata=(
                    buy_signals[["AI_Confidence"]]
                    if "AI_Confidence" in buy_signals.columns
                    else None
                ),
            ),
            row=1,
            col=1,
        )

    # Add sell signals
    sell_signals = data[data["Signal"] == -1]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals["Date"],
                y=sell_signals["Close"],
                mode="markers",
                name="Sell Signal",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="red",
                    line=dict(color="darkred", width=2),
                ),
                hovertemplate="<b>AI SELL SIGNAL</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>AI Confidence: %{customdata[0]:.1%}<extra></extra>",
                customdata=(
                    sell_signals[["AI_Confidence"]]
                    if "AI_Confidence" in sell_signals.columns
                    else None
                ),
            ),
            row=1,
            col=1,
        )

    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=data["Date"],
            y=data["Volume"],
            name="Volume",
            marker_color="rgba(158, 158, 158, 0.5)",
            hovertemplate="<b>Volume</b><br>Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode="x unified",
        xaxis2_title="Date",
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


def display_integrated_ai_analysis(data: pd.DataFrame, ticker: str) -> None:
    """Integrated AI trading analysis with comprehensive insights.

    This function combines trading signals overview and AI analysis insights
    into a unified, comprehensive dashboard showing AI decision-making process.

    Args:
        data (pd.DataFrame): The processed data containing AI analysis results.
        ticker (str): The stock ticker symbol for context.
    """
    # Check if AI analysis data is available
    if "AI_Signal" not in data.columns or "AI_Rationale" not in data.columns:
        st.warning(
            "⚠️ AI analysis data not available. Please ensure API keys are configured correctly."
        )
        return

    st.header(f"🤖 AI Trading Analysis for {ticker}")

    # === OVERVIEW METRICS ===
    analyzed_days = data[
        (data["AI_Signal"] != "No analysis performed")
        & (data["AI_Rationale"] != "No analysis performed")
        & (data["AI_Confidence"] > 0)
    ].copy()

    no_analysis_days = data[
        (data["AI_Signal"] == "No analysis performed")
        | (data["AI_Rationale"] == "No analysis performed")
        | (data["AI_Confidence"] == 0)
    ].copy()

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Total Days", len(data))
    with col2:
        st.metric("🤖 Days with AI Analysis", len(analyzed_days))
    with col3:
        if analyzed_days.empty:
            st.metric("📊 Signal Performance", "No signals")
        else:
            buy_count = len(analyzed_days[analyzed_days["Signal"] == 1])
            sell_count = len(analyzed_days[analyzed_days["Signal"] == -1])
            st.metric("📊 Signal Performance", f"{buy_count}B/{sell_count}S")
    with col4:
        if analyzed_days.empty:
            st.metric("🎯 Avg Confidence", "N/A")
        else:
            avg_confidence = analyzed_days["AI_Confidence"].mean()
            st.metric("🎯 Avg Confidence", f"{avg_confidence:.1%}")

    # === AI ANALYSIS RESULTS ===
    if analyzed_days.empty:
        st.info("ℹ️ No AI analysis performed in this date range.")
        st.markdown(
            """
        **📊 Possible reasons:**
        - No relevant news found for this stock in the selected date range
        - Try a more recent date range (last 30 days work best)
        - Popular stocks like AAPL, TSLA, MSFT typically have more news coverage
        - Check API quotas - Gemini Free Tier has limited daily requests
        """
        )
        return

    st.subheader("📊 AI Analysis Results")

    # === LATEST AI DECISION (HIGHLIGHTED) ===
    latest_analysis = analyzed_days.iloc[-1]

    # Calculate daily performance for latest analysis
    daily_change = latest_analysis["Close"] - latest_analysis["Open"]
    daily_change_pct = (daily_change / latest_analysis["Open"]) * 100

    with st.container():
        st.markdown("#### 🔥 Latest AI Decision")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

        with col1:
            signal_color = (
                "green"
                if latest_analysis["Signal"] == 1
                else "red" if latest_analysis["Signal"] == -1 else "orange"
            )
            signal_emoji = (
                "🟢"
                if latest_analysis["Signal"] == 1
                else "🔴" if latest_analysis["Signal"] == -1 else "🟡"
            )
            st.markdown(f"**{signal_emoji} {latest_analysis['AI_Signal']}**")
            st.write(f"**Date:** {latest_analysis['Date']}")

        with col2:
            confidence_color = (
                "green"
                if latest_analysis["AI_Confidence"] > 0.7
                else "orange" if latest_analysis["AI_Confidence"] > 0.5 else "red"
            )
            st.markdown(
                f"**Confidence:** <span style='color: {confidence_color}'>{latest_analysis['AI_Confidence']:.1%}</span>",
                unsafe_allow_html=True,
            )

        with col3:
            # Price information with trend
            price_color = (
                "green"
                if daily_change_pct > 0
                else "red" if daily_change_pct < 0 else "gray"
            )
            trend_emoji = (
                "📈"
                if daily_change_pct > 0.5
                else "📉" if daily_change_pct < -0.5 else "➖"
            )
            st.write("**Stock Performance:**")
            st.write(f"Open: ${latest_analysis['Open']:.2f}")
            st.write(f"Close: ${latest_analysis['Close']:.2f}")
            st.markdown(
                f"**{trend_emoji} <span style='color: {price_color}'>{daily_change_pct:+.1f}%</span>**",
                unsafe_allow_html=True,
            )

        with col4:
            st.write("**AI Reasoning:**")
            st.info(
                latest_analysis["AI_Rationale"]
                if pd.notna(latest_analysis["AI_Rationale"])
                else "No rationale available"
            )

    # === DETAILED ANALYSIS TABLE ===
    if len(analyzed_days) > 1:
        st.subheader("📋 Complete AI Analysis History")

        # Prepare display data with comprehensive price information
        display_data = analyzed_days.copy()
        display_data["AI_Decision"] = display_data["Signal"].map(
            {1: "🟢 BUY", -1: "🔴 SELL", 0: "🟡 HOLD"}
        )

        # Calculate daily price change and percentage
        display_data["Daily_Change"] = display_data["Close"] - display_data["Open"]
        display_data["Daily_Change_Pct"] = (
            display_data["Daily_Change"] / display_data["Open"] * 100
        )

        # Add price change emoji indicator
        display_data["Price_Trend"] = display_data["Daily_Change_Pct"].apply(
            lambda x: "📈" if x > 0.5 else "📉" if x < -0.5 else "➖"
        )

        # Select and format columns (added comprehensive price info)
        display_columns = [
            "Date",
            "Open",
            "Close",
            "High",
            "Low",
            "Daily_Change_Pct",
            "Price_Trend",
            "AI_Decision",
            "AI_Confidence",
        ]
        table_data = display_data[display_columns].copy()

        # Rename columns
        table_data.columns = [
            "Date",
            "Open ($)",
            "Close ($)",
            "High ($)",
            "Low ($)",
            "Change (%)",
            "Trend",
            "AI Decision",
            "Confidence",
        ]

        # Format data
        table_data["Open ($)"] = table_data["Open ($)"].round(2)
        table_data["Close ($)"] = table_data["Close ($)"].round(2)
        table_data["High ($)"] = table_data["High ($)"].round(2)
        table_data["Low ($)"] = table_data["Low ($)"].round(2)
        table_data["Change (%)"] = table_data["Change (%)"].round(2).astype(str) + "%"
        table_data["Confidence"] = (table_data["Confidence"] * 100).round(1).astype(
            str
        ) + "%"

        # Sort by date (most recent first)
        table_data = table_data.sort_values("Date", ascending=False)

        st.dataframe(table_data, use_container_width=True, hide_index=True)

        # === EXPANDABLE AI REASONING SECTION ===
        st.subheader("🧠 Detailed AI Reasoning")
        st.markdown("*Click to expand reasoning for each analysis:*")

        # Sort analyzed_days for consistent ordering
        reasoning_data = analyzed_days.sort_values("Date", ascending=False)

        for _, row in reasoning_data.iterrows():
            date_str = (
                row["Date"].strftime("%Y-%m-%d")
                if hasattr(row["Date"], "strftime")
                else str(row["Date"])
            )
            signal_emoji = (
                "🟢" if row["Signal"] == 1 else "🔴" if row["Signal"] == -1 else "🟡"
            )
            signal_text = row["AI_Signal"]
            confidence = row["AI_Confidence"]

            # Calculate day's price performance
            day_change = row["Close"] - row["Open"]
            day_change_pct = (day_change / row["Open"]) * 100
            price_trend_emoji = (
                "📈"
                if day_change_pct > 0.5
                else "📉" if day_change_pct < -0.5 else "➖"
            )

            # Create expandable section for each day's reasoning
            with st.expander(
                f"{signal_emoji} {date_str} - {signal_text} (Confidence: {confidence:.1%}) | {price_trend_emoji} {day_change_pct:+.1f}%",
                expanded=False,
            ):
                st.write("**Full AI Analysis:**")
                st.info(
                    row["AI_Rationale"]
                    if pd.notna(row["AI_Rationale"])
                    else "No detailed reasoning available"
                )

                # Comprehensive context with price data
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Trading Day:**")
                    st.write(f"📅 Date: {date_str}")
                    st.write(f"🤖 AI Decision: {signal_text}")
                    st.write(f"🎯 Confidence: {confidence:.1%}")

                with col2:
                    st.write("**Price Details:**")
                    st.write(f"🔓 Open: ${row['Open']:.2f}")
                    st.write(f"🔒 Close: ${row['Close']:.2f}")
                    st.write(f"⬆️ High: ${row['High']:.2f}")
                    st.write(f"⬇️ Low: ${row['Low']:.2f}")

                with col3:
                    st.write("**Performance:**")
                    price_color = (
                        "green"
                        if day_change_pct > 0
                        else "red" if day_change_pct < 0 else "gray"
                    )
                    st.markdown(
                        f"📊 Change: <span style='color: {price_color}'>${day_change:+.2f}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"{price_trend_emoji} Percent: <span style='color: {price_color}'>{day_change_pct:+.1f}%</span>",
                        unsafe_allow_html=True,
                    )

                    # Volume if available
                    if "Volume" in row and pd.notna(row["Volume"]):
                        st.write(f"📊 Volume: {row['Volume']:,.0f}")

                    # AI vs Reality comparison
                    ai_expectation = (
                        "� Bullish"
                        if row["Signal"] == 1
                        else "� Bearish" if row["Signal"] == -1 else "➖ Neutral"
                    )
                    reality = (
                        "📈 Up"
                        if day_change_pct > 0.1
                        else "📉 Down" if day_change_pct < -0.1 else "➖ Flat"
                    )
                    match_color = (
                        "green"
                        if (row["Signal"] == 1 and day_change_pct > 0.1)
                        or (row["Signal"] == -1 and day_change_pct < -0.1)
                        or (row["Signal"] == 0 and abs(day_change_pct) <= 0.5)
                        else "orange"
                    )
                    st.markdown(f"🔮 AI Expected: {ai_expectation}")
                    st.markdown(f"📈 Reality: {reality}")
                    st.markdown(
                        f"✓ <span style='color: {match_color}'>{'Match' if match_color == 'green' else 'Mismatch'}</span>",
                        unsafe_allow_html=True,
                    )

    # === PERFORMANCE SUMMARY ===
    st.subheader("📈 AI Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    buy_signals = len(analyzed_days[analyzed_days["Signal"] == 1])
    sell_signals = len(analyzed_days[analyzed_days["Signal"] == -1])
    hold_signals = len(analyzed_days[analyzed_days["Signal"] == 0])

    with col1:
        st.metric("🟢 Buy Signals", buy_signals)
    with col2:
        st.metric("🔴 Sell Signals", sell_signals)
    with col3:
        st.metric("🟡 Hold Signals", hold_signals)
    with col4:
        high_confidence_count = len(analyzed_days[analyzed_days["AI_Confidence"] > 0.7])
        st.metric("🎯 High Confidence", f"{high_confidence_count}/{len(analyzed_days)}")

    # === AI DECISION PATTERNS ===
    if len(analyzed_days) > 2:
        st.subheader("🧠 AI Decision Patterns")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Signal Distribution:**")
            signal_dist = analyzed_days["AI_Signal"].value_counts()
            for signal_type, count in signal_dist.items():
                percentage = (count / len(analyzed_days)) * 100
                st.write(f"• {signal_type}: {count} ({percentage:.1f}%)")

        with col2:
            st.write("**Confidence Analysis:**")
            high_conf = len(analyzed_days[analyzed_days["AI_Confidence"] > 0.7])
            med_conf = len(
                analyzed_days[
                    (analyzed_days["AI_Confidence"] >= 0.5)
                    & (analyzed_days["AI_Confidence"] <= 0.7)
                ]
            )
            low_conf = len(analyzed_days[analyzed_days["AI_Confidence"] < 0.5])

            st.write(f"• High (>70%): {high_conf} decisions")
            st.write(f"• Medium (50-70%): {med_conf} decisions")
            st.write(f"• Low (<50%): {low_conf} decisions")


# This function has been integrated into display_integrated_ai_analysis()
# Kept as placeholder for backward compatibility


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

            if signal_summary["total_days"] > 0:
                signal_rate = (
                    signal_summary["total_signals"] / signal_summary["total_days"]
                ) * 100
                st.write(f"• Signal Rate: {signal_rate:.1f}%")

            # Add AI-specific statistics if available
            if "analyses_performed" in signal_summary:
                st.write(
                    f"• AI Analyses Performed: {signal_summary['analyses_performed']}"
                )
            if (
                "avg_confidence" in signal_summary
                and signal_summary["avg_confidence"] > 0
            ):
                st.write(
                    f"• Average AI Confidence: {signal_summary['avg_confidence']:.1%}"
                )
        else:
            st.info("No signals generated for statistical analysis.")

    # Additional AI insights
    st.subheader("🤖 AI Analysis Summary")

    # Show AI analysis status
    latest_data = data.iloc[-1]

    if "AI_Signal" in latest_data and pd.notna(latest_data["AI_Signal"]):
        ai_position = latest_data["AI_Signal"]
        confidence = latest_data.get("AI_Confidence", 0)

        st.write(f"**Current AI Position:** {ai_position}")
        if confidence > 0:
            st.write(f"• AI Confidence: {confidence:.1%}")

        # Show recent AI signal if any
        recent_signals = data[data["Signal"] != 0].tail(1)
        if not recent_signals.empty:
            last_signal = recent_signals.iloc[-1]
            signal_type = last_signal.get("AI_Signal", "UNKNOWN")
            confidence = last_signal.get("AI_Confidence", 0)
            st.write(
                f"**Most Recent AI Signal:** {signal_type} on {last_signal['Date'].strftime('%Y-%m-%d')}"
            )
            if confidence > 0:
                st.write(f"• Signal Confidence: {confidence:.1%}")
    else:
        st.info("AI analysis data not available for the current dataset.")


def display_realtime_monitoring(data: pd.DataFrame, ticker: str) -> None:
    """Display real-time monitoring interface integrated from NiceGUI functionality.

    This function provides real-time price updates, live signal monitoring,
    and interactive dashboard features within the Streamlit interface.

    Args:
        data (pd.DataFrame): Historical data with AI signals.
        ticker (str): Stock ticker symbol.
    """
    st.subheader(f"⚡ Real-time Monitor for {ticker}")

    # Check if we have data to monitor
    if data.empty:
        st.warning(
            "⚠️ No historical data available for real-time monitoring. Please load historical data first."
        )
        return

    # Initialize session state for real-time monitoring
    if "monitoring_active" not in st.session_state:
        st.session_state.monitoring_active = False
    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = None
    if "current_price_data" not in st.session_state:
        st.session_state.current_price_data = None
    if "realtime_data_manager" not in st.session_state:
        config = get_config()
        st.session_state.realtime_data_manager = DataManager(
            cache_duration=config.realtime_cache_duration
        )

    # Control panel
    with st.container():
        st.markdown("### 🎛️ Control Panel")

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            # Start/Stop monitoring button
            if st.button(
                (
                    "🔴 Stop Monitor"
                    if st.session_state.monitoring_active
                    else "🟢 Start Monitor"
                ),
                type="secondary" if st.session_state.monitoring_active else "primary",
            ):
                st.session_state.monitoring_active = (
                    not st.session_state.monitoring_active
                )
                if st.session_state.monitoring_active:
                    st.success(f"✅ Started monitoring {ticker}")
                    st.rerun()
                else:
                    st.info("⏸️ Stopped monitoring")
                    st.rerun()

        with col2:
            # Auto-refresh toggle
            auto_refresh = st.checkbox(
                "Auto-refresh (30s)", value=st.session_state.monitoring_active
            )
            if auto_refresh and st.session_state.monitoring_active:
                time.sleep(1)  # Small delay to prevent too frequent updates
                st.rerun()

        with col3:
            # Last update time display
            if st.session_state.last_update_time:
                st.write(f"**Last Update:** {st.session_state.last_update_time}")
            else:
                st.write("**Status:** Not monitoring")

    # Real-time metrics row
    if st.session_state.monitoring_active:
        try:
            # Fetch latest price data
            latest_price_info = st.session_state.realtime_data_manager.get_latest_price(
                ticker
            )

            if latest_price_info:
                st.session_state.current_price_data = latest_price_info
                st.session_state.last_update_time = latest_price_info[
                    "timestamp"
                ].strftime("%H:%M:%S")

                # Display current metrics
                with st.container():
                    st.markdown("### 📊 Current Metrics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Current Price",
                            f"${latest_price_info['price']:.2f}",
                            delta=f"{latest_price_info['change']:+.2f} ({latest_price_info['change_percent']:+.2f}%)",
                        )

                    with col2:
                        # Get latest AI signal from historical data
                        latest_signal_data = data.iloc[-1] if not data.empty else None
                        if (
                            latest_signal_data is not None
                            and "AI_Signal" in data.columns
                        ):
                            signal_value = latest_signal_data.get("AI_Signal", "HOLD")
                            signal_emoji = (
                                "🟢"
                                if signal_value == "BUY"
                                else "🔴" if signal_value == "SELL" else "🟡"
                            )
                            st.metric(
                                "Latest AI Signal", f"{signal_emoji} {signal_value}"
                            )
                        else:
                            st.metric("Latest AI Signal", "🟡 HOLD")

                    with col3:
                        if (
                            latest_signal_data is not None
                            and "AI_Confidence" in data.columns
                        ):
                            confidence = latest_signal_data.get("AI_Confidence", 0.0)
                            st.metric("AI Confidence", f"{confidence:.1%}")
                        else:
                            st.metric("AI Confidence", "N/A")

                    with col4:
                        # Volume information
                        volume = latest_price_info.get("volume", 0)
                        if volume > 0:
                            volume_str = f"{volume:,.0f}"
                        else:
                            volume_str = "N/A"
                        st.metric("Volume", volume_str)

                # Live chart with recent data
                display_realtime_chart(data, latest_price_info, ticker)

                # Recent signals table (last 5)
                display_recent_signals_compact(data, ticker)

            else:
                st.error(
                    f"❌ Failed to fetch real-time data for {ticker}. Please check the ticker symbol."
                )

        except Exception as e:
            error_message = str(e)
            if (
                "RESOURCE_EXHAUSTED" in error_message
                or "quota" in error_message.lower()
            ):
                st.error("🚨 **API Quota Exhausted** - Real-time monitoring limited")
                st.info(
                    "💡 Price data can still be updated, but AI analysis features are temporarily unavailable. Will reset at UTC midnight tomorrow."
                )
            else:
                st.error(f"❌ Error in real-time monitoring: {error_message}")

    else:
        # Show static overview when not monitoring
        st.info(
            "🔍 Click 'Start Monitor' to begin real-time price tracking and live AI signal updates."
        )

        # Show latest data from historical analysis
        if not data.empty:
            with st.container():
                st.markdown("### 📈 Latest Historical Data")
                latest_row = data.iloc[-1]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Last Close Price", f"${latest_row['Close']:.2f}")
                with col2:
                    if "AI_Signal" in data.columns:
                        signal = latest_row.get("AI_Signal", "HOLD")
                        signal_emoji = (
                            "🟢"
                            if signal == "BUY"
                            else "🔴" if signal == "SELL" else "🟡"
                        )
                        st.metric("Last AI Signal", f"{signal_emoji} {signal}")
                with col3:
                    if "AI_Confidence" in data.columns:
                        confidence = latest_row.get("AI_Confidence", 0.0)
                        st.metric("Signal Confidence", f"{confidence:.1%}")


def display_realtime_chart(data: pd.DataFrame, price_info: Dict, ticker: str) -> None:
    """Display real-time price chart with latest price point.

    Args:
        data (pd.DataFrame): Historical data with AI signals.
        price_info (Dict): Latest price information.
        ticker (str): Stock ticker symbol.
    """
    try:
        # Create chart with recent data (last 30 days)
        recent_data = data.tail(30).copy() if len(data) > 30 else data.copy()

        fig = go.Figure()

        # Add historical price line
        fig.add_trace(
            go.Scatter(
                x=recent_data["Date"],
                y=recent_data["Close"],
                mode="lines",
                name="Historical Price",
                line=dict(color="#1f77b4", width=2),
            )
        )

        # Add current price point (use last date + 1 day as estimate)
        if not recent_data.empty:
            last_date = recent_data["Date"].iloc[-1]
            current_date = last_date + pd.Timedelta(days=1)

            fig.add_trace(
                go.Scatter(
                    x=[current_date],
                    y=[price_info["price"]],
                    mode="markers",
                    name="Current Price",
                    marker=dict(size=12, color="red", symbol="circle"),
                )
            )

        # Add buy signals
        buy_signals = recent_data[recent_data.get("Signal", 0) == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals["Date"],
                    y=buy_signals["Close"],
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(symbol="triangle-up", size=10, color="green"),
                )
            )

        # Add sell signals
        sell_signals = recent_data[recent_data.get("Signal", 0) == -1]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals["Date"],
                    y=sell_signals["Close"],
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(symbol="triangle-down", size=10, color="red"),
                )
            )

        # Update layout
        fig.update_layout(
            title=f"{ticker} - Real-time Price with AI Signals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
            showlegend=True,
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating real-time chart: {str(e)}")


def display_recent_signals_compact(data: pd.DataFrame, ticker: str) -> None:
    """Display recent AI signals in a compact table.

    Args:
        data (pd.DataFrame): Data with AI signals.
        ticker (str): Stock ticker symbol.
    """
    try:
        if data.empty or "Signal" not in data.columns:
            return

        # Get signals (non-zero only)
        signals_data = data[data["Signal"] != 0].tail(5)  # Last 5 signals

        if signals_data.empty:
            st.info("🔍 No recent trading signals found in historical data.")
            return

        st.markdown("### 📋 Recent AI Signals")

        # Create table data
        table_data = []
        for _, row in signals_data.iterrows():
            signal_emoji = "🟢" if row["Signal"] == 1 else "🔴"
            signal_text = row.get("AI_Signal", "BUY" if row["Signal"] == 1 else "SELL")

            table_data.append(
                {
                    "Date": (
                        row["Date"].strftime("%Y-%m-%d")
                        if hasattr(row["Date"], "strftime")
                        else str(row["Date"])
                    ),
                    "Signal": f"{signal_emoji} {signal_text}",
                    "Price": f"${row['Close']:.2f}",
                    "Confidence": f"{row.get('AI_Confidence', 0.0):.1%}",
                    "Rationale": (
                        str(row.get("AI_Rationale", "N/A"))[:50] + "..."
                        if len(str(row.get("AI_Rationale", "N/A"))) > 50
                        else str(row.get("AI_Rationale", "N/A"))
                    ),
                }
            )

        # Display as dataframe
        df_display = pd.DataFrame(table_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error displaying recent signals: {str(e)}")


def display_api_status_check() -> None:
    """Display API status information and quota check in the sidebar."""
    with st.sidebar.expander("🔌 API Status", expanded=False):
        try:
            # Load configuration to check API keys
            config = get_config()
            api_validation = config.validate_api_keys()

            # Check Gemini AI API
            if api_validation.get("google_api_key", False):
                st.success("✅ Gemini AI API Key: Configured")

                # Try to get AI analyzer status
                try:
                    from core.ai_analyzer import get_ai_analyzer

                    ai_analyzer = get_ai_analyzer()
                    ai_stats = ai_analyzer.get_analysis_stats()

                    st.write("**🤖 AI Settings:**")
                    st.write(f"• Model: {ai_stats['model_name']}")
                    st.write(f"• Rate Limit: {ai_stats['rate_limit_requests']}/min")

                    # Test connection (lightweight)
                    with st.spinner("Testing AI connection..."):
                        connection_test = ai_analyzer.test_connection()
                        if connection_test.get("connected", False):
                            st.success("🟢 AI Service: Online")
                        else:
                            error_msg = connection_test.get("error", "Unknown error")
                            if (
                                "RESOURCE_EXHAUSTED" in str(error_msg)
                                or "quota" in str(error_msg).lower()
                            ):
                                st.error("🔴 AI Service: Quota Exhausted")
                                st.warning(
                                    "⚠️ Daily quota of 200 requests exhausted, resets at UTC midnight"
                                )
                            else:
                                st.warning(f"🟡 AI Service: {error_msg}")
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                        st.error("🔴 API Quota Exhausted")
                        st.info("💡 Demo mode available")
                    else:
                        st.warning(f"🟡 AI Status: {str(e)[:50]}...")
            else:
                st.error("❌ Gemini AI API Key: Not configured")

            # Check News API
            if api_validation.get("newsapi_api_key", False):
                st.success("✅ News API Key: Configured")

                try:
                    from core.news_fetcher import get_news_fetcher

                    news_fetcher = get_news_fetcher()
                    news_status = news_fetcher.validate_connection()

                    if news_status.get("connected", False):
                        st.success("🟢 News Service: Online")
                        sources_count = news_status.get("sources_available", 0)
                        if sources_count > 0:
                            st.write(f"• Sources: {sources_count} available")
                    else:
                        error_msg = news_status.get("error", "Unknown error")
                        st.warning(f"🟡 News Service: {error_msg}")
                except Exception as e:
                    st.warning(f"🟡 News Status: {str(e)[:50]}...")
            else:
                st.error("❌ News API Key: Not configured")

            # Show demo mode availability
            st.divider()
            st.info(
                "💡 **Demo Mode**: When API quota is exhausted, the application automatically switches to demo mode displaying simulated trading signals."
            )

        except Exception as e:
            st.error(f"Status check error: {str(e)}")


def display_ai_model_selection() -> None:
    """Display AI model selection interface in the sidebar."""
    with st.sidebar.expander("🤖 AI Model Selection", expanded=False):
        try:
            config = get_config()

            # Get available models
            available_models = config.get_available_models()

            st.write("**Select AI Model:**")

            # Create options for selectbox
            model_options = {}
            for model_id, model_data in available_models.items():
                display_name = f"{model_data['name']} ({model_data['tier']})"
                model_options[display_name] = model_id

            # Find current selection index
            current_display_name = None
            for display_name, model_id in model_options.items():
                if model_id == config.ai_model_name:
                    current_display_name = display_name
                    break

            # Model selection
            selected_display_name = st.selectbox(
                "Choose AI Model",
                options=list(model_options.keys()),
                index=(
                    list(model_options.keys()).index(current_display_name)
                    if current_display_name
                    else 0
                ),
                help="Different models have different capabilities and rate limits",
            )

            selected_model_id = model_options[selected_display_name]

            # Display model information
            if selected_model_id in available_models:
                model_info = available_models[selected_model_id]

                st.write("**Model Details:**")
                st.write(f"• **Description**: {model_info['description']}")
                st.write(
                    f"• **Rate Limit**: {model_info['rate_limit_requests']} requests/min"
                )
                st.write(f"• **Max Tokens**: {model_info['max_tokens']:,}")
                st.write(f"• **Tier**: {model_info['tier'].title()}")

                # Show source of model information
                model_source = model_info.get("source", "hardcoded")
                if model_source == "api":
                    st.success("🔄 **Source**: Live from Google API")
                else:
                    st.info("📋 **Source**: Hardcoded fallback")

                # Add refresh button for dynamic models
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "🔄 Refresh Models", help="Fetch latest models from Google API"
                    ):
                        with st.spinner("Fetching models from API..."):
                            try:
                                success = config.refresh_available_models()
                                if success:
                                    st.success("✅ Models refreshed successfully")
                                    st.rerun()
                                else:
                                    st.warning(
                                        "⚠️ Could not fetch from API, using fallback models"
                                    )
                            except Exception as e:
                                st.error(f"❌ Error refreshing models: {str(e)}")

                # Update model if changed
                with col2:
                    if selected_model_id != config.ai_model_name:
                        if st.button(
                            "🔄 Update Model", help="Apply the selected AI model"
                        ):
                            with st.spinner("Updating AI model..."):
                                try:
                                    # Update configuration
                                    success = config.update_model_settings(
                                        selected_model_id
                                    )

                                    if success:
                                        # Update AI analyzer
                                        from core.ai_analyzer import (
                                            get_ai_analyzer,
                                        )

                                        ai_analyzer = get_ai_analyzer()
                                        ai_analyzer.update_model(selected_model_id)

                                        st.success(
                                            f"✅ Updated to {model_info['name']}"
                                        )
                                        st.info(
                                            "ℹ️ New model will be used for the next analysis"
                                        )

                                        # Clear any cached data to force refresh
                                        if "data_loaded" in st.session_state:
                                            del st.session_state["data_loaded"]

                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to update model")

                                except Exception as e:
                                    st.error(f"❌ Error updating model: {str(e)}")

            # Information about different models
            st.divider()
            st.info(
                """
            **💡 Model Differences:**
            
            • **Flash Lite**: Fastest responses, lowest cost, basic reasoning
            • **Flash**: Balanced speed and capability
            • **Pro**: Advanced reasoning, highest capability, slower responses
            
            **Rate Limits**: Different models have different rate limits. The application automatically adjusts based on your selection.
            """
            )

        except Exception as e:
            st.error(f"Model selection error: {str(e)}")


if __name__ == "__main__":
    main()
