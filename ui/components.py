"""
UI Components for AI Trading Demo

This module provides all UI components and utilities for the Streamlit application,
combining display components, message utilities, and refactored functions into a
unified interface.

Author: AI Trading Demo Team
Version: 3.0 (Reorganized Architecture)
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# UI message types
class MessageType(Enum):
    """Types of UI messages."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class UIMessage:
    """Structured UI message container."""
    
    message_type: MessageType
    title: str
    content: str
    icon: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "type": self.message_type.value,
            "title": self.title,
            "content": self.content,
            "icon": self.icon,
            "details": self.details
        }


def display_streamlit_message(
    message_type: MessageType,
    title: str,
    content: str,
    icon: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Display formatted message in Streamlit interface.
    
    Args:
        message_type (MessageType): Type of message to display
        title (str): Message title
        content (str): Message content
        icon (Optional[str]): Icon to display with message
        details (Optional[Dict[str, Any]]): Additional details to show
    """
    if icon:
        title = f"{icon} {title}"
    
    if message_type == MessageType.SUCCESS:
        st.success(f"**{title}**\n\n{content}")
    elif message_type == MessageType.ERROR:
        st.error(f"**{title}**\n\n{content}")
    elif message_type == MessageType.WARNING:
        st.warning(f"**{title}**\n\n{content}")
    else:
        st.info(f"**{title}**\n\n{content}")
    
    if details:
        with st.expander("Details"):
            for key, value in details.items():
                st.write(f"**{key}:** {value}")


# Display Components
def display_ai_overview_metrics(data: pd.DataFrame, analyzed_days: pd.DataFrame) -> None:
    """Display the top-level AI analysis overview metrics.
    
    Args:
        data (pd.DataFrame): Full dataset
        analyzed_days (pd.DataFrame): Days with AI analysis performed
    """
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


def display_no_analysis_message() -> None:
    """Display message when no AI analysis is available."""
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


def get_signal_display_info(signal_value: int) -> Tuple[str, str]:
    """Get display color and emoji for a signal value.
    
    Args:
        signal_value (int): Signal value (1, -1, or 0)
        
    Returns:
        Tuple[str, str]: (color, emoji)
    """
    if signal_value == 1:
        return "green", "🟢"
    elif signal_value == -1:
        return "red", "🔴"
    else:
        return "orange", "🟡"


def get_confidence_color(confidence: float) -> str:
    """Get color for confidence display based on confidence level.
    
    Args:
        confidence (float): Confidence value (0.0 to 1.0)
        
    Returns:
        str: Color name for display
    """
    if confidence > 0.7:
        return "green"
    elif confidence > 0.5:
        return "orange"
    else:
        return "red"


def calculate_price_performance(row: pd.Series) -> Tuple[float, float, str, str]:
    """Calculate price performance metrics for a trading day.
    
    Args:
        row (pd.Series): Row from DataFrame with OHLC data
        
    Returns:
        Tuple[float, float, str, str]: (daily_change, daily_change_pct, price_color, trend_emoji)
    """
    daily_change = row["Close"] - row["Open"]
    daily_change_pct = (daily_change / row["Open"]) * 100
    
    if daily_change_pct > 0:
        price_color = "green"
    elif daily_change_pct < 0:
        price_color = "red"
    else:
        price_color = "gray"
    
    if daily_change_pct > 0.5:
        trend_emoji = "📈"
    elif daily_change_pct < -0.5:
        trend_emoji = "📉"
    else:
        trend_emoji = "➖"
    
    return daily_change, daily_change_pct, price_color, trend_emoji


def display_latest_ai_decision(latest_analysis: pd.Series) -> None:
    """Display the latest AI decision in a highlighted format.
    
    Args:
        latest_analysis (pd.Series): Latest analysis row from data
    """
    # Calculate performance metrics
    daily_change, daily_change_pct, price_color, trend_emoji = calculate_price_performance(
        latest_analysis
    )
    
    st.markdown("#### 🔥 Latest AI Decision")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        signal_color, signal_emoji = get_signal_display_info(latest_analysis["Signal"])
        st.markdown(f"**{signal_emoji} {latest_analysis['AI_Signal']}**")
        st.write(f"**Date:** {latest_analysis['Date']}")
    
    with col2:
        confidence_color = get_confidence_color(latest_analysis["AI_Confidence"])
        st.markdown(
            f"**Confidence:** <span style='color: {confidence_color}'>{latest_analysis['AI_Confidence']:.1%}</span>",
            unsafe_allow_html=True,
        )
    
    with col3:
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


def prepare_analysis_table_data(analyzed_days: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for the AI analysis history table.
    
    Args:
        analyzed_days (pd.DataFrame): Days with AI analysis data
        
    Returns:
        pd.DataFrame: Formatted table data
    """
    display_data = analyzed_days.copy()
    
    # Add signal display column
    display_data["AI_Decision"] = display_data["Signal"].map(
        {1: "🟢 BUY", -1: "🔴 SELL", 0: "🟡 HOLD"}
    )
    
    # Calculate daily price changes
    display_data["Daily_Change"] = display_data["Close"] - display_data["Open"]
    display_data["Daily_Change_Pct"] = (
        display_data["Daily_Change"] / display_data["Open"] * 100
    )
    
    # Add price trend indicators
    display_data["Price_Trend"] = display_data["Daily_Change_Pct"].apply(
        lambda x: "📈" if x > 0.5 else "📉" if x < -0.5 else "➖"
    )
    
    # Select and format columns
    display_columns = [
        "Date", "Open", "Close", "High", "Low", 
        "Daily_Change_Pct", "Price_Trend", "AI_Decision", "AI_Confidence"
    ]
    table_data = display_data[display_columns].copy()
    
    # Rename columns for display
    table_data.columns = [
        "Date", "Open ($)", "Close ($)", "High ($)", "Low ($)",
        "Change (%)", "Trend", "AI Decision", "Confidence"
    ]
    
    # Format numerical columns
    for col in ["Open ($)", "Close ($)", "High ($)", "Low ($)"]:
        table_data[col] = table_data[col].round(2)
    
    table_data["Change (%)"] = table_data["Change (%)"].round(2).astype(str) + "%"
    table_data["Confidence"] = (table_data["Confidence"] * 100).round(1).astype(str) + "%"
    
    # Sort by date (most recent first)
    return table_data.sort_values("Date", ascending=False)


def display_analysis_history_table(analyzed_days: pd.DataFrame) -> None:
    """Display the complete AI analysis history table.
    
    Args:
        analyzed_days (pd.DataFrame): Days with AI analysis performed
    """
    if len(analyzed_days) <= 1:
        return
    
    st.subheader("📋 Complete AI Analysis History")
    
    # Prepare and display table
    table_data = prepare_analysis_table_data(analyzed_days)
    st.dataframe(table_data, use_container_width=True, hide_index=True)


def display_ai_reasoning_expandable(analyzed_days: pd.DataFrame) -> None:
    """Display expandable AI reasoning sections for each analysis day.
    
    Args:
        analyzed_days (pd.DataFrame): Days with AI analysis performed
    """
    if len(analyzed_days) <= 1:
        return
    
    st.subheader("🧠 Detailed AI Reasoning")
    st.markdown("*Click to expand reasoning for each analysis:*")
    
    # Sort data for consistent ordering
    reasoning_data = analyzed_days.sort_values("Date", ascending=False)
    
    for _, row in reasoning_data.iterrows():
        _display_single_reasoning_expander(row)


def _display_single_reasoning_expander(row: pd.Series) -> None:
    """Display a single expandable reasoning section.
    
    Args:
        row (pd.Series): Single row of analysis data
    """
    # Format date
    date_str = (
        row["Date"].strftime("%Y-%m-%d")
        if hasattr(row["Date"], "strftime")
        else str(row["Date"])
    )
    
    # Get display elements
    signal_color, signal_emoji = get_signal_display_info(row["Signal"])
    signal_text = row["AI_Signal"]
    confidence = row["AI_Confidence"]
    
    # Calculate performance
    daily_change, daily_change_pct, price_color, trend_emoji = calculate_price_performance(row)
    
    # Create expandable section
    with st.expander(
        f"{signal_emoji} {date_str} - {signal_text} (Confidence: {confidence:.1%}) | {trend_emoji} {daily_change_pct:+.1f}%",
        expanded=False,
    ):
        st.write("**Full AI Analysis:**")
        st.info(
            row["AI_Rationale"]
            if pd.notna(row["AI_Rationale"])
            else "No detailed reasoning available"
        )
        
        # Display detailed context in columns
        _display_reasoning_context(row, date_str, signal_text, confidence, 
                                 daily_change, daily_change_pct, price_color, trend_emoji)


def _display_reasoning_context(row: pd.Series, date_str: str, signal_text: str, 
                              confidence: float, daily_change: float, 
                              daily_change_pct: float, price_color: str, trend_emoji: str) -> None:
    """Display detailed context for reasoning expandable section.
    
    Args:
        row (pd.Series): Analysis row data
        date_str (str): Formatted date string
        signal_text (str): AI signal text
        confidence (float): AI confidence level
        daily_change (float): Daily price change in dollars
        daily_change_pct (float): Daily price change percentage
        price_color (str): Color for price display
        trend_emoji (str): Emoji for trend display
    """
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
        st.markdown(
            f"📊 Change: <span style='color: {price_color}'>${daily_change:+.2f}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{trend_emoji} Percent: <span style='color: {price_color}'>{daily_change_pct:+.1f}%</span>",
            unsafe_allow_html=True,
        )
        
        # Volume if available
        if "Volume" in row and pd.notna(row["Volume"]):
            st.write(f"📊 Volume: {row['Volume']:,.0f}")
        
        # AI vs Reality comparison
        _display_ai_vs_reality_comparison(row, daily_change_pct)


def _display_ai_vs_reality_comparison(row: pd.Series, daily_change_pct: float) -> None:
    """Display AI expectation vs reality comparison.
    
    Args:
        row (pd.Series): Analysis row data
        daily_change_pct (float): Daily price change percentage
    """
    # AI expectation
    if row["Signal"] == 1:
        ai_expectation = "🐂 Bullish"
    elif row["Signal"] == -1:
        ai_expectation = "🐻 Bearish"
    else:
        ai_expectation = "➖ Neutral"
    
    # Reality
    if daily_change_pct > 0.1:
        reality = "📈 Up"
    elif daily_change_pct < -0.1:
        reality = "📉 Down"
    else:
        reality = "➖ Flat"
    
    # Match assessment
    is_match = (
        (row["Signal"] == 1 and daily_change_pct > 0.1) or
        (row["Signal"] == -1 and daily_change_pct < -0.1) or
        (row["Signal"] == 0 and abs(daily_change_pct) <= 0.5)
    )
    match_color = "green" if is_match else "orange"
    match_text = "Match" if is_match else "Mismatch"
    
    st.markdown(f"🔮 AI Expected: {ai_expectation}")
    st.markdown(f"📈 Reality: {reality}")
    st.markdown(
        f"✓ <span style='color: {match_color}'>{match_text}</span>",
        unsafe_allow_html=True,
    )


def display_performance_summary(analyzed_days: pd.DataFrame) -> None:
    """Display AI performance summary metrics.
    
    Args:
        analyzed_days (pd.DataFrame): Days with AI analysis performed
    """
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


def display_decision_patterns(analyzed_days: pd.DataFrame) -> None:
    """Display AI decision patterns analysis.
    
    Args:
        analyzed_days (pd.DataFrame): Days with AI analysis performed
    """
    if len(analyzed_days) <= 2:
        return
    
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
                (analyzed_days["AI_Confidence"] >= 0.5) &
                (analyzed_days["AI_Confidence"] <= 0.7)
            ]
        )
        low_conf = len(analyzed_days[analyzed_days["AI_Confidence"] < 0.5])
        
        st.write(f"• High (>70%): {high_conf} decisions")
        st.write(f"• Medium (50-70%): {med_conf} decisions")
        st.write(f"• Low (<50%): {low_conf} decisions")


# Main refactored function
def display_integrated_ai_analysis(data: pd.DataFrame, ticker: str) -> None:
    """Integrated AI trading analysis with comprehensive insights.

    This function combines trading signals overview and AI analysis insights
    into a unified, comprehensive dashboard showing AI decision-making process.
    Uses modular UI components for better maintainability.

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

    # Filter analyzed days
    analyzed_days = data[
        (data["AI_Signal"] != "No analysis performed")
        & (data["AI_Rationale"] != "No analysis performed")
        & (data["AI_Confidence"] > 0)
    ].copy()

    # Display overview metrics
    display_ai_overview_metrics(data, analyzed_days)

    # Handle case with no analysis
    if analyzed_days.empty:
        display_no_analysis_message()
        return

    st.subheader("📊 AI Analysis Results")

    # Display latest AI decision (highlighted)
    latest_analysis = analyzed_days.iloc[-1]
    with st.container():
        display_latest_ai_decision(latest_analysis)

    # Display analysis history table
    display_analysis_history_table(analyzed_days)

    # Display expandable AI reasoning sections
    display_ai_reasoning_expandable(analyzed_days)

    # Display performance summary
    display_performance_summary(analyzed_days)

    # Display decision patterns
    display_decision_patterns(analyzed_days)