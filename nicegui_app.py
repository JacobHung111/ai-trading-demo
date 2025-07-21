"""
AI Trading Demo - NiceGUI Real-time Monitoring Application

This is the real-time monitoring application built with NiceGUI. It provides
live price updates, real-time signal generation, and an interactive dashboard
for active trading monitoring and quick decision making.

Author: AI Trading Demo Team
Version: 1.0 (Hybrid Architecture)
"""

import datetime
import asyncio
from typing import Optional, Dict, Any
import pandas as pd
from nicegui import ui, app, run
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our shared components
from shared.config import get_config
from shared.data_manager import DataManager
from shared.strategy import TradingStrategy
from shared.indicators import TechnicalIndicators


class TradingDashboard:
    """Real-time trading dashboard with NiceGUI interface."""

    def __init__(self):
        """Initialize the trading dashboard."""
        config = get_config()
        self.data_manager = DataManager(cache_duration=config.realtime_cache_duration)
        self.strategy = TradingStrategy(
            short_window=config.sma_short_period, long_window=config.sma_long_period
        )
        self.current_ticker = config.default_ticker
        self.config = config
        self.is_monitoring = False
        self.current_data: Optional[pd.DataFrame] = None
        self.update_timer: Optional[ui.timer] = None
        
        # User configurable settings
        self.data_period_days = 150  # Default data period

        # UI components references
        self.price_card = None
        self.signal_card = None
        self.chart_container = None
        self.status_indicator = None
        self.ticker_input = None
        self.monitor_button = None
        self.last_update_time = None
        self.main_container = None

    def setup_ui(self) -> None:
        """Set up the user interface layout and components."""
        # Set up the page
        ui.page_title("AI Trading Demo - Real-time Monitor")

        with ui.header():
            with ui.row().classes("w-full justify-between items-center"):
                ui.label("📈 AI Trading Monitor").classes("text-xl font-bold")
                with ui.row().classes("items-center gap-4"):
                    self.status_indicator = ui.badge("Offline", color="red")
                    self.last_update_time = ui.label("Last Update: Never").classes(
                        "text-sm text-gray-400"
                    )
                    ui.button("Settings", icon="settings", on_click=self.show_settings)

        # Main dashboard layout
        self.main_container = ui.column().classes("w-full max-w-7xl mx-auto p-4 gap-4")
        with self.main_container:

            # Control panel
            with ui.card().classes("w-full"):
                ui.label("Control Panel").classes("text-lg font-semibold mb-2")
                with ui.row().classes("w-full gap-4 items-end"):
                    self.ticker_input = ui.input(
                        "Stock Ticker",
                        value=self.current_ticker,
                        validation={"Invalid ticker": lambda value: len(value) > 0},
                    ).classes("flex-grow")

                    self.monitor_button = ui.button(
                        "Start Monitoring",
                        color="green",
                        on_click=self.toggle_monitoring,
                    )

                    ui.button(
                        "Load Historical",
                        color="blue",
                        on_click=self.load_historical_data,
                    )

            # Key metrics row
            with ui.row().classes("w-full gap-4"):
                # Current price card
                with ui.card().classes("flex-1"):
                    ui.label("Current Price").classes("text-sm text-gray-600")
                    self.price_card = ui.label("$0.00").classes(
                        "text-2xl font-bold text-blue-600"
                    )
                    self.price_change = ui.label("+$0.00 (0.00%)").classes("text-sm")

                # Latest signal card
                with ui.card().classes("flex-1"):
                    ui.label("Latest Signal").classes("text-sm text-gray-600")
                    self.signal_card = ui.label("No Signal").classes(
                        "text-2xl font-bold"
                    )
                    self.signal_time = ui.label("--").classes("text-sm text-gray-600")

                # SMA values card
                with ui.card().classes("flex-1"):
                    ui.label("Moving Averages").classes("text-sm text-gray-600")
                    self.sma20_label = ui.label("SMA20: $0.00").classes("text-sm")
                    self.sma50_label = ui.label("SMA50: $0.00").classes("text-sm")

            # Chart section
            with ui.card().classes("w-full"):
                ui.label("Price Chart with Signals").classes(
                    "text-lg font-semibold mb-2"
                )
                self.chart_container = ui.column().classes("w-full h-96")

            # Recent signals table
            with ui.card().classes("w-full"):
                ui.label("Recent Signals").classes("text-lg font-semibold mb-2")
                self.signals_table = ui.table(
                    columns=[
                        {"name": "time", "label": "Time", "field": "time"},
                        {"name": "signal", "label": "Signal", "field": "signal"},
                        {"name": "price", "label": "Price", "field": "price"},
                        {"name": "sma20", "label": "SMA20", "field": "sma20"},
                        {"name": "sma50", "label": "SMA50", "field": "sma50"},
                    ],
                    rows=[],
                ).classes("w-full")

    async def toggle_monitoring(self) -> None:
        """Toggle real-time monitoring on/off."""
        if not self.is_monitoring:
            # Start monitoring
            ticker = self.ticker_input.value.strip().upper()
            if not ticker:
                ui.notify("Please enter a valid ticker symbol", type="negative")
                return

            if not self.data_manager.validate_ticker(ticker):
                ui.notify(f"Invalid ticker symbol: {ticker}", type="negative")
                return

            # Check if historical data is loaded
            if self.current_data is None or self.current_data.empty:
                ui.notify("⚠️ Please load historical data first before starting monitoring", type="warning")
                return

            self.current_ticker = ticker
            self.is_monitoring = True
            self.monitor_button.props("color=red")
            self.monitor_button.text = "Stop Monitoring"
            self.status_indicator.props("color=green")
            self.status_indicator.text = "Online"

            # Start the monitoring timer
            self.update_timer = ui.timer(
                10.0, self.update_live_data_sync
            )  # Update every 10 seconds
            ui.notify(f"Started monitoring {ticker}", type="positive")

        else:
            # Stop monitoring
            self.is_monitoring = False
            if self.update_timer:
                self.update_timer.cancel()

            self.monitor_button.props("color=green")
            self.monitor_button.text = "Start Monitoring"
            self.status_indicator.props("color=red")
            self.status_indicator.text = "Offline"
            ui.notify("Stopped monitoring", type="info")

    def update_live_data_sync(self) -> None:
        """Synchronous wrapper for live data updates using ui.timer."""
        if not self.is_monitoring:
            return

        try:
            # Update timestamp to show activity
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            self.last_update_time.text = f"Last Update: {current_time}"

            # Schedule the async update
            asyncio.create_task(self.async_price_update())

        except Exception as e:
            print(f"Error in sync update: {e}")
            ui.notify(f"Update error: {str(e)}", type="negative")

    async def async_price_update(self) -> None:
        """Async helper for price updates."""
        try:
            # Get latest price info
            price_info = await self.data_manager.get_latest_price_async(
                self.current_ticker
            )
            if price_info:
                # Use main container context for UI updates
                with self.main_container:
                    self.update_price_display(price_info)
                    ui.notify(
                        f'Updated {self.current_ticker} price: ${price_info["price"]:.2f}',
                        type="info",
                        timeout=3000,
                    )

                    # Update historical data with latest price and recalculate signals
                    if self.current_data is not None and not self.current_data.empty:
                        self.update_realtime_data(price_info)
                        self.update_chart()
                        self.update_signal_display()
                        self.update_signals_table()

        except Exception as e:
            print(f"Error in async price update: {e}")

    def update_realtime_data(self, price_info: Dict) -> None:
        """Update the last row of historical data with real-time price and recalculate signals."""
        try:
            if self.current_data is None or self.current_data.empty:
                return

            # Update the last row with new price data
            latest_price = price_info.get("price", 0)
            self.current_data.loc[self.current_data.index[-1], "Close"] = latest_price
            
            # Recalculate indicators for the updated data
            from shared.indicators import TechnicalIndicators
            self.current_data = TechnicalIndicators.add_all_indicators(self.current_data)
            
            # Recalculate signals
            self.current_data = self.strategy.generate_signals(self.current_data)
            
        except Exception as e:
            print(f"Error updating realtime data: {e}")

    def update_price_display(self, price_info: Dict) -> None:
        """Update the price display with latest information."""
        price = price_info.get("price", 0)
        change = price_info.get("change", 0)
        change_percent = price_info.get("change_percent", 0)

        self.price_card.text = f"${price:.2f}"

        # Color code the change
        sign = "+" if change >= 0 else ""
        self.price_change.text = f"{sign}${change:.2f} ({change_percent:.2f}%)"
        # Update classes using the style method for NiceGUI
        self.price_change.style(f'color: {"green" if change >= 0 else "red"}')

    async def load_historical_data(self) -> None:
        """Load historical data for analysis."""
        ticker = self.ticker_input.value.strip().upper()
        if not ticker:
            ui.notify("Please enter a valid ticker symbol", type="negative")
            return

        # Use configurable data period
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=self.data_period_days)

        ui.notify("Loading historical data...", type="info")

        try:
            # Fetch data
            raw_data = await self.data_manager.fetch_stock_data_async(
                ticker, start_date, end_date
            )

            if raw_data.empty:
                ui.notify("No data found for the specified period", type="negative")
                return

            # Calculate indicators and signals
            data_with_indicators = TechnicalIndicators.add_all_indicators(raw_data)
            
            # Check if we have sufficient data for SMA50
            sma50_valid = data_with_indicators['SMA50'].notna().sum()
            if sma50_valid < 10:  # Need at least 10 valid SMA50 values for meaningful signals
                ui.notify(f"⚠️ Insufficient data for reliable signals. Only {sma50_valid} valid SMA50 values found. Try a different ticker or longer period.", type="warning")
                return
            
            self.current_data = self.strategy.generate_signals(data_with_indicators)

            if self.current_data.empty:
                ui.notify("Insufficient data for analysis", type="negative")
                return

            # Check if we found any signals
            if 'Signal' in self.current_data.columns:
                total_signals = len(self.current_data[self.current_data['Signal'] != 0])
                if total_signals == 0:
                    ui.notify("ℹ️ No trading signals found in current data period. This is normal for stable markets.", type="info")
                else:
                    ui.notify(f"✅ Found {total_signals} trading signals in historical data", type="positive")

            # Update UI
            self.update_chart()
            self.update_signal_display()
            self.update_signals_table()
            ui.notify("Historical data loaded successfully", type="positive")

        except Exception as e:
            ui.notify(f"Error loading data: {str(e)}", type="negative")

    def update_chart(self) -> None:
        """Update the price chart with current data."""
        if self.current_data is None or self.current_data.empty:
            return

        try:
            # Ensure we're in the correct UI context
            # Create the chart
            fig = make_subplots(
                rows=1,
                cols=1,
                subplot_titles=[f"{self.current_ticker} Price with SMA Signals"],
            )

            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=self.current_data["Date"],
                    y=self.current_data["Close"],
                    mode="lines",
                    name="Close Price",
                    line=dict(color="#2563eb", width=2),
                )
            )

            # Add SMA lines
            fig.add_trace(
                go.Scatter(
                    x=self.current_data["Date"],
                    y=self.current_data["SMA20"],
                    mode="lines",
                    name="SMA20",
                    line=dict(color="#f59e0b", width=1),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=self.current_data["Date"],
                    y=self.current_data["SMA50"],
                    mode="lines",
                    name="SMA50",
                    line=dict(color="#10b981", width=1),
                )
            )

            # Add buy signals
            buy_signals = self.current_data[self.current_data["Signal"] == 1]
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
            sell_signals = self.current_data[self.current_data["Signal"] == -1]
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
                height=350,
                margin=dict(l=50, r=50, t=50, b=50),
                showlegend=True,
                template="plotly_white",
            )

            # Update the chart container - use ui.plotly for NiceGUI
            # Clear and rebuild chart in the correct context
            self.chart_container.clear()
            with self.chart_container:
                ui.plotly(fig).classes("w-full h-96")

        except Exception as e:
            print(f"Error updating chart: {e}")

    def update_signal_display(self) -> None:
        """Update the latest signal display."""
        if self.current_data is None or self.current_data.empty:
            return

        try:
            latest_signal = self.strategy.get_latest_signal(self.current_data)

            if latest_signal:
                signal_type = latest_signal["type"]

                # Safely update UI components
                if self.signal_card:
                    self.signal_card.text = f"{signal_type} Signal"
                    # Update style using the style method for NiceGUI
                    signal_color = "green" if signal_type == "BUY" else "red"
                    self.signal_card.style(
                        f"color: {signal_color}; font-weight: bold; font-size: 1.5rem"
                    )

                signal_date = latest_signal["date"]
                if hasattr(signal_date, "strftime"):
                    date_str = signal_date.strftime("%Y-%m-%d")
                else:
                    date_str = str(signal_date)

                if self.signal_time:
                    self.signal_time.text = f"Date: {date_str}"

                # Update SMA values
                if self.sma20_label:
                    self.sma20_label.text = f"SMA20: ${latest_signal['sma20']:.2f}"
                if self.sma50_label:
                    self.sma50_label.text = f"SMA50: ${latest_signal['sma50']:.2f}"
            else:
                if self.signal_card:
                    self.signal_card.text = "No Signal"
                    self.signal_card.style(
                        "color: gray; font-weight: bold; font-size: 1.5rem"
                    )
                if self.signal_time:
                    self.signal_time.text = "--"

        except Exception as e:
            print(f"Error updating signal display: {e}")

    def update_signals_table(self) -> None:
        """Update the recent signals table."""
        if self.current_data is None or self.current_data.empty:
            return

        try:
            signal_list = self.strategy.get_signal_list(self.current_data)

            # Take only the last 10 signals
            recent_signals = signal_list[-10:] if len(signal_list) > 10 else signal_list

            # Format for table
            table_rows = []
            for signal in reversed(recent_signals):  # Most recent first
                date = signal["date"]
                if hasattr(date, "strftime"):
                    date_str = date.strftime("%Y-%m-%d")
                else:
                    date_str = str(date)

                table_rows.append(
                    {
                        "time": date_str,
                        "signal": signal["type"],
                        "price": f"${signal['price']:.2f}",
                        "sma20": f"${signal['sma20']:.2f}",
                        "sma50": f"${signal['sma50']:.2f}",
                    }
                )

            self.signals_table.rows = table_rows

        except Exception as e:
            print(f"Error updating signals table: {e}")

    def check_for_new_signals(self) -> None:
        """Check for new signals and send notifications."""
        # This would be implemented for real-time signal detection
        # For now, we'll just update the display
        self.update_signal_display()

    def show_settings(self) -> None:
        """Show settings dialog."""
        with ui.dialog() as dialog, ui.card().classes("w-[500px] max-w-[90vw]"):
            ui.label("⚙️ Settings").classes("text-xl font-bold mb-6 text-center")

            with ui.column().classes("gap-6"):
                # Data Settings
                with ui.column().classes("gap-3"):
                    ui.label("📊 Data Settings").classes("text-lg font-semibold text-blue-600")
                    data_period_input = ui.input(
                        "Historical Data Period (days)", 
                        value=str(self.data_period_days),
                        validation={"Must be between 60-365": lambda value: 60 <= int(value or 0) <= 365}
                    ).props("type=number min=60 max=365").classes("w-full")
                    
                    ui.label("💡 More days = better signal accuracy, but slower loading").classes("text-sm text-gray-500 pl-2")
                
                # Strategy Settings  
                ui.separator()
                with ui.column().classes("gap-3"):
                    ui.label("📈 Strategy Settings").classes("text-lg font-semibold text-green-600")
                    
                    with ui.row().classes("gap-4 w-full"):
                        sma_short_input = ui.input(
                            "SMA Short Period", 
                            value=str(self.config.sma_short_period),
                            validation={"Must be 5-30": lambda value: 5 <= int(value or 0) <= 30}
                        ).props("type=number min=5 max=30").classes("flex-1")
                        
                        sma_long_input = ui.input(
                            "SMA Long Period", 
                            value=str(self.config.sma_long_period),
                            validation={"Must be 30-100": lambda value: 30 <= int(value or 0) <= 100}
                        ).props("type=number min=30 max=100").classes("flex-1")
                    
                    ui.label("📌 SMA crossover strategy: Buy when short crosses above long").classes("text-sm text-gray-500 pl-2")
                
                # Real-time Settings (Future)
                ui.separator()
                with ui.column().classes("gap-3"):
                    ui.label("🔄 Real-time Settings").classes("text-lg font-semibold text-gray-400")
                    
                    with ui.row().classes("gap-4 w-full"):
                        ui.input(
                            "Update Interval (sec)", 
                            value="10",
                            validation={"Must be 5-60": lambda value: 5 <= int(value or 0) <= 60}
                        ).props("type=number min=5 max=60 disable").classes("flex-1 opacity-40")
                        
                        ui.input(
                            "Cache Duration (sec)",
                            value=str(self.config.realtime_cache_duration),
                            validation={"Must be 30-300": lambda value: 30 <= int(value or 0) <= 300}
                        ).props("type=number min=30 max=300 disable").classes("flex-1 opacity-40")
                    
                    ui.switch("Enable Notifications", value=True).props("disable").classes("opacity-40")
                    
                    ui.label("ℹ️ Real-time settings will be available in future updates").classes("text-sm text-gray-400 pl-2")

                def save_settings():
                    try:
                        settings_changed = False
                        strategy_changed = False
                        
                        # Update data period
                        new_period = int(data_period_input.value)
                        if 60 <= new_period <= 365 and new_period != self.data_period_days:
                            self.data_period_days = new_period
                            settings_changed = True
                            
                        # Update strategy settings
                        new_short = int(sma_short_input.value)
                        new_long = int(sma_long_input.value)
                        if (5 <= new_short <= 30 and 30 <= new_long <= 100 and new_short < new_long and
                            (new_short != self.strategy.short_window or new_long != self.strategy.long_window)):
                            self.strategy = TradingStrategy(short_window=new_short, long_window=new_long)
                            strategy_changed = True
                            
                        if settings_changed or strategy_changed:
                            ui.notify("✅ Settings saved! Please reload historical data to apply changes.", type="positive")
                            # Clear current data to force reload
                            self.current_data = None
                        else:
                            ui.notify("ℹ️ No changes to save.", type="info")
                            
                        dialog.close()
                        
                    except ValueError:
                        ui.notify("❌ Please enter valid numbers", type="negative")

                # Action buttons
                ui.separator()
                with ui.row().classes("gap-3 justify-end mt-6"):
                    ui.button("Cancel", color="red", on_click=dialog.close).props("outline")
                    ui.button("💾 Save Settings", color="green", on_click=save_settings)

        dialog.open()


def main() -> None:
    """Main application entry point."""
    dashboard = TradingDashboard()
    dashboard.setup_ui()

    # Set up the app (no static files needed for this demo)

    # Run the application
    ui.run(
        title="AI Trading Demo - Real-time Monitor",
        port=8081,  # Changed port to avoid conflicts
        reload=False,
        show=False,  # Auto-open browser for user interaction
    )


if __name__ == "__main__":
    main()
