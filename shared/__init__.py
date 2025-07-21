"""
Shared Components Package for AI Trading Demo

This package contains modular components shared between the Streamlit and NiceGUI
applications, providing centralized data management, trading strategy logic,
and technical indicators calculation.

Modules:
    data_manager: Centralized data fetching and caching
    strategy: Core trading strategy logic (SMA crossover)
    indicators: Technical indicators calculation (SMA20, SMA50)

Author: AI Trading Demo Team
Version: 1.0 (Hybrid Architecture)
"""

from .data_manager import DataManager
from .strategy import TradingStrategy
from .indicators import TechnicalIndicators

__all__ = ["DataManager", "TradingStrategy", "TechnicalIndicators"]

__version__ = "1.0"
