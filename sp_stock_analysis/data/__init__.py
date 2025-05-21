"""
Data module for SP stock analysis.
This package handles data loading, processing, and management for stock analysis.
"""

from .fetchers import (
    get_sp500_symbols,
    get_historical_data,
    fetch_with_retry,
    fetch_batch,
    validate_data_structure
)

from .news import NewsAnalyzer

__all__ = [
    # Data fetching functions
    'get_sp500_symbols',
    'get_historical_data',
    'fetch_with_retry',
    'fetch_batch',
    'validate_data_structure',
    
    # News analysis
    'NewsAnalyzer',
]

__version__ = '0.1.0'
