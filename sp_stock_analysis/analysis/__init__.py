"""
Analysis module for SP stock analysis.
Provides technical and sentiment analysis tools for stock market data.
"""

from .technical import (
    calculate_rsi,
    calculate_macd,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_obv,
    calculate_stochastic,
    calculate_adx,
    calculate_technical_indicators,
    ensure_series
)

from .sentiment import (
    calculate_flexible_technical_sentiment,
    combine_sentiment_signals,
    get_sentiment_recommendation,
    analyze_stock_sentiment
)

__all__ = [
    # Technical analysis functions
    'calculate_rsi',
    'calculate_macd',
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_obv',
    'calculate_stochastic',
    'calculate_adx',
    'calculate_technical_indicators',
    'ensure_series',
    
    # Sentiment analysis functions
    'calculate_flexible_technical_sentiment',
    'combine_sentiment_signals',
    'get_sentiment_recommendation',
    'analyze_stock_sentiment'
]

__version__ = '0.1.0' 