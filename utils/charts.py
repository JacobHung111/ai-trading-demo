"""
Chart Utilities for AI Trading Demo

This module provides centralized chart creation utilities and visualization patterns
shared across all chart components in the application.

Author: AI Trading Demo Team
Version: 1.0 (Refactored for Code Deduplication)
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.config import get_config


class ChartConfig:
    """Standard chart configuration and theming."""
    
    def __init__(self):
        """Initialize chart configuration."""
        config = get_config()
        self.height = config.chart_height
        self.theme = config.chart_theme
        
        # Color scheme
        self.colors = {
            "buy_signal": "#00D4AA",  # Green
            "sell_signal": "#FF6B6B",  # Red
            "hold_signal": "#4ECDC4",  # Teal
            "price_line": "#2E86C1",  # Blue
            "sma20": "#F39C12",  # Orange
            "sma50": "#8E44AD",  # Purple
            "volume": "#95A5A6",  # Gray
            "background": "#FFFFFF",  # White
            "grid": "#E5E5E5"  # Light gray
        }
        
        # Chart layout defaults
        self.layout_defaults = {
            "height": self.height,
            "template": self.theme,
            "showlegend": True,
            "hovermode": "x unified",
            "xaxis": {"showgrid": True, "gridcolor": self.colors["grid"]},
            "yaxis": {"showgrid": True, "gridcolor": self.colors["grid"]},
            "plot_bgcolor": self.colors["background"],
            "paper_bgcolor": self.colors["background"]
        }


def create_price_chart_with_signals(
    data: pd.DataFrame,
    ticker: str,
    show_volume: bool = True,
    show_sma: bool = True,
    height: Optional[int] = None
) -> go.Figure:
    """Create comprehensive price chart with trading signals and indicators.
    
    Args:
        data (pd.DataFrame): Stock data with signals.
        ticker (str): Stock ticker symbol.
        show_volume (bool): Whether to show volume subplot.
        show_sma (bool): Whether to show SMA lines.
        height (Optional[int]): Chart height override.
        
    Returns:
        go.Figure: Configured plotly figure.
    """
    config = ChartConfig()
    
    # Create subplots based on requirements
    subplot_specs = [[{"secondary_y": False}]]
    if show_volume:
        subplot_specs.append([{"secondary_y": False}])
    
    fig = make_subplots(
        rows=len(subplot_specs),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f"{ticker} Price & Trading Signals"] + (["Volume"] if show_volume else []),
        specs=subplot_specs,
        row_heights=[0.7, 0.3] if show_volume else [1.0]
    )
    
    # Add candlestick/price line
    fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add SMA lines if requested
    if show_sma and "SMA20" in data.columns and "SMA50" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["SMA20"],
                mode="lines",
                name="SMA 20",
                line=dict(color=config.colors["sma20"], width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["SMA50"],
                mode="lines",
                name="SMA 50",
                line=dict(color=config.colors["sma50"], width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Add trading signals
    _add_trading_signals_to_chart(fig, data, config, row=1)
    
    # Add AI signals if available
    if "AI_Signal" in data.columns:
        _add_ai_signals_to_chart(fig, data, config, row=1)
    
    # Add volume if requested
    if show_volume and "Volume" in data.columns:
        fig.add_trace(
            go.Bar(
                x=data["Date"],
                y=data["Volume"],
                name="Volume",
                marker_color=config.colors["volume"],
                opacity=0.6
            ),
            row=2, col=1
        )
    
    # Update layout
    layout_update = config.layout_defaults.copy()
    if height:
        layout_update["height"] = height
    
    layout_update.update({
        "title": f"{ticker} Trading Analysis",
        "xaxis_title": "Date",
        "yaxis_title": "Price ($)",
    })
    
    if show_volume:
        layout_update["yaxis2_title"] = "Volume"
    
    fig.update_layout(**layout_update)
    
    return fig


def create_ai_analysis_summary_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Create AI analysis summary visualization.
    
    Args:
        data (pd.DataFrame): Data with AI analysis results.
        ticker (str): Stock ticker symbol.
        
    Returns:
        go.Figure: AI analysis summary chart.
    """
    config = ChartConfig()
    
    # Filter data with AI analysis
    ai_data = data[
        (data.get("AI_Signal", "") != "No analysis performed") &
        (data.get("AI_Confidence", 0) > 0)
    ].copy()
    
    if ai_data.empty:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No AI analysis data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(**config.layout_defaults, title=f"AI Analysis Summary - {ticker}")
        return fig
    
    # Create confidence distribution chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "AI Signal Distribution",
            "Confidence Over Time", 
            "Confidence Distribution",
            "Signal Performance"
        ],
        specs=[[{"type": "pie"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    # Signal distribution pie chart
    signal_counts = ai_data["AI_Signal"].value_counts()
    fig.add_trace(
        go.Pie(
            labels=signal_counts.index,
            values=signal_counts.values,
            name="AI Signals",
            marker_colors=[
                config.colors["buy_signal"] if label == "BUY" 
                else config.colors["sell_signal"] if label == "SELL"
                else config.colors["hold_signal"]
                for label in signal_counts.index
            ]
        ),
        row=1, col=1
    )
    
    # Confidence over time
    fig.add_trace(
        go.Scatter(
            x=ai_data["Date"],
            y=ai_data["AI_Confidence"],
            mode="lines+markers",
            name="AI Confidence",
            line=dict(color=config.colors["price_line"])
        ),
        row=1, col=2
    )
    
    # Confidence distribution histogram
    fig.add_trace(
        go.Histogram(
            x=ai_data["AI_Confidence"],
            nbinsx=10,
            name="Confidence Distribution",
            marker_color=config.colors["price_line"],
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Signal performance (if we have traditional signals to compare)
    if "Signal" in ai_data.columns:
        # Compare AI signals vs traditional signals
        comparison_data = []
        for signal_type in ["BUY", "SELL", "HOLD"]:
            ai_count = len(ai_data[ai_data["AI_Signal"] == signal_type])
            trad_signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
            trad_count = len(ai_data[ai_data["Signal"] == trad_signal_map[signal_type]])
            comparison_data.extend([
                {"Signal": signal_type, "Type": "AI", "Count": ai_count},
                {"Signal": signal_type, "Type": "Traditional", "Count": trad_count}
            ])
        
        comparison_df = pd.DataFrame(comparison_data)
        
        for signal_type in ["AI", "Traditional"]:
            type_data = comparison_df[comparison_df["Type"] == signal_type]
            fig.add_trace(
                go.Bar(
                    x=type_data["Signal"],
                    y=type_data["Count"],
                    name=signal_type,
                    opacity=0.8
                ),
                row=2, col=2
            )
    
    # Update layout
    fig.update_layout(
        height=600,
        template=config.theme,
        title_text=f"AI Analysis Summary - {ticker}",
        showlegend=True
    )
    
    return fig


def create_realtime_price_chart(
    historical_data: pd.DataFrame,
    current_price: float,
    price_change: float,
    ticker: str,
    time_window: int = 30
) -> go.Figure:
    """Create real-time price monitoring chart.
    
    Args:
        historical_data (pd.DataFrame): Historical price data.
        current_price (float): Current/latest price.
        price_change (float): Price change amount.
        ticker (str): Stock ticker symbol.
        time_window (int): Number of recent periods to show.
        
    Returns:
        go.Figure: Real-time price chart.
    """
    config = ChartConfig()
    
    # Get recent data
    recent_data = historical_data.tail(time_window).copy()
    
    # Add current price point
    if not recent_data.empty:
        latest_date = recent_data["Date"].iloc[-1]
        if isinstance(latest_date, str):
            latest_date = pd.to_datetime(latest_date).date()
        
        # Create synthetic current point
        current_point = pd.DataFrame({
            "Date": [latest_date],
            "Close": [current_price]
        })
        
        # Combine data
        display_data = pd.concat([recent_data[["Date", "Close"]], current_point], ignore_index=True)
    else:
        display_data = pd.DataFrame({
            "Date": [datetime.date.today()],
            "Close": [current_price]
        })
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    color = config.colors["buy_signal"] if price_change >= 0 else config.colors["sell_signal"]
    
    fig.add_trace(
        go.Scatter(
            x=display_data["Date"],
            y=display_data["Close"],
            mode="lines+markers",
            name=f"{ticker} Price",
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color)
        )
    )
    
    # Highlight current price
    if len(display_data) > 1:
        fig.add_trace(
            go.Scatter(
                x=[display_data["Date"].iloc[-1]],
                y=[current_price],
                mode="markers",
                name="Current Price",
                marker=dict(
                    size=12,
                    color=color,
                    symbol="diamond",
                    line=dict(width=2, color="white")
                ),
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        **config.layout_defaults,
        title=f"{ticker} Real-time Price: ${current_price:.2f} ({price_change:+.2f})",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=300
    )
    
    return fig


def _add_trading_signals_to_chart(
    fig: go.Figure,
    data: pd.DataFrame,
    config: ChartConfig,
    row: int = 1
) -> None:
    """Add traditional trading signals to chart.
    
    Args:
        fig (go.Figure): Plotly figure to modify.
        data (pd.DataFrame): Data with trading signals.
        config (ChartConfig): Chart configuration.
        row (int): Subplot row number.
    """
    if "Signal" not in data.columns:
        return
    
    # Buy signals
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
                    color=config.colors["buy_signal"],
                    line=dict(width=2, color="white")
                ),
                hovertemplate="<b>BUY SIGNAL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
            ),
            row=row, col=1
        )
    
    # Sell signals
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
                    color=config.colors["sell_signal"],
                    line=dict(width=2, color="white")
                ),
                hovertemplate="<b>SELL SIGNAL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
            ),
            row=row, col=1
        )


def _add_ai_signals_to_chart(
    fig: go.Figure,
    data: pd.DataFrame,
    config: ChartConfig,
    row: int = 1
) -> None:
    """Add AI trading signals to chart.
    
    Args:
        fig (go.Figure): Plotly figure to modify.
        data (pd.DataFrame): Data with AI signals.
        config (ChartConfig): Chart configuration.
        row (int): Subplot row number.
    """
    if "AI_Signal" not in data.columns:
        return
    
    # AI Buy signals
    ai_buy_signals = data[data["AI_Signal"] == "BUY"]
    if not ai_buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=ai_buy_signals["Date"],
                y=ai_buy_signals["Close"],
                mode="markers",
                name="AI Buy",
                marker=dict(
                    symbol="star",
                    size=10,
                    color=config.colors["buy_signal"],
                    line=dict(width=1, color="white")
                ),
                hovertemplate="<b>AI BUY</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>Confidence: %{customdata:.1%}<extra></extra>",
                customdata=ai_buy_signals.get("AI_Confidence", [])
            ),
            row=row, col=1
        )
    
    # AI Sell signals
    ai_sell_signals = data[data["AI_Signal"] == "SELL"]
    if not ai_sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=ai_sell_signals["Date"],
                y=ai_sell_signals["Close"],
                mode="markers",
                name="AI Sell",
                marker=dict(
                    symbol="star",
                    size=10,
                    color=config.colors["sell_signal"],
                    line=dict(width=1, color="white")
                ),
                hovertemplate="<b>AI SELL</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>Confidence: %{customdata:.1%}<extra></extra>",
                customdata=ai_sell_signals.get("AI_Confidence", [])
            ),
            row=row, col=1
        )


def create_signal_comparison_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Create chart comparing traditional vs AI signals.
    
    Args:
        data (pd.DataFrame): Data with both signal types.
        ticker (str): Stock ticker symbol.
        
    Returns:
        go.Figure: Signal comparison chart.
    """
    config = ChartConfig()
    
    # Filter data with both signal types
    comparison_data = data[
        (data.get("Signal", 0) != 0) | 
        (data.get("AI_Signal", "") != "No analysis performed")
    ].copy()
    
    if comparison_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No signal data available for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(**config.layout_defaults, title=f"Signal Comparison - {ticker}")
        return fig
    
    # Create comparison chart
    fig = go.Figure()
    
    # Add price line as background
    fig.add_trace(
        go.Scatter(
            x=comparison_data["Date"],
            y=comparison_data["Close"],
            mode="lines",
            name="Price",
            line=dict(color=config.colors["price_line"], width=1),
            opacity=0.5
        )
    )
    
    # Add traditional signals
    _add_trading_signals_to_chart(fig, comparison_data, config)
    
    # Add AI signals
    _add_ai_signals_to_chart(fig, comparison_data, config)
    
    # Update layout
    fig.update_layout(
        **config.layout_defaults,
        title=f"Traditional vs AI Signals - {ticker}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500
    )
    
    return fig


def format_chart_for_export(fig: go.Figure, format_type: str = "png") -> go.Figure:
    """Format chart for export with appropriate styling.
    
    Args:
        fig (go.Figure): Chart to format.
        format_type (str): Export format (png, pdf, svg).
        
    Returns:
        go.Figure: Formatted chart.
    """
    # Clone figure to avoid modifying original
    export_fig = go.Figure(fig)
    
    # Update styling for export
    export_fig.update_layout(
        font=dict(size=12),
        title_font_size=16,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return export_fig