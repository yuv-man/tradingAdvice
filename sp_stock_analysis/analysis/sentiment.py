"""
Sentiment analysis module for stock analysis.
Combines technical and news sentiment analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from ..data import NewsAnalyzer

def calculate_flexible_technical_sentiment(df: pd.DataFrame) -> pd.Series:
    """Calculate a sentiment score based on technical indicators"""
    sentiment = pd.Series(0.0, index=df.index)
    
    try:
        # RSI component (-0.4 to 0.4)
        rsi_sentiment = pd.Series(0.0, index=df.index)
        rsi_sentiment[df['RSI'] < 30] = 0.4  # Oversold - positive sentiment
        rsi_sentiment[(df['RSI'] >= 30) & (df['RSI'] <= 70)] = (df['RSI'] - 50) / 50 * 0.2
        rsi_sentiment[df['RSI'] > 70] = -0.4  # Overbought - negative sentiment
        
        # MACD component (-0.2 to 0.2)
        macd_sentiment = pd.Series(0.0, index=df.index)
        macd_sentiment[df['MACD'] > df['Signal_Line']] = 0.2
        macd_sentiment[df['MACD'] <= df['Signal_Line']] = -0.2
        
        # Moving Average component (-0.2 to 0.2)
        ma_sentiment = pd.Series(0.0, index=df.index)
        ma_bullish = ((df['Close'] > df['SMA20']) & 
                     (df['Close'] > df['SMA50']) & 
                     (df['SMA20'] > df['SMA50'])).astype(int)
        ma_bearish = ((df['Close'] < df['SMA20']) & 
                     (df['Close'] < df['SMA50']) & 
                     (df['SMA20'] < df['SMA50'])).astype(int)
        ma_sentiment = (ma_bullish - ma_bearish) * 0.2
        
        # Bollinger Bands component (-0.1 to 0.1)
        bb_sentiment = pd.Series(0.0, index=df.index)
        bb_sentiment[df['Close'] < df['BB_Lower']] = 0.1
        bb_sentiment[df['Close'] > df['BB_Upper']] = -0.1
        
        # Volume component (-0.1 to 0.1)
        vol_sentiment = pd.Series(0.0, index=df.index)
        vol_sentiment[df['Volume_Ratio'] > 1.5] = 0.1 * np.sign(df['Daily_Return'])
        
        # Combine all components
        sentiment = (rsi_sentiment + 
                    macd_sentiment + 
                    ma_sentiment + 
                    bb_sentiment + 
                    vol_sentiment)
        
        # Normalize to [-1, 1] range
        sentiment = sentiment.clip(-1, 1)
        
    except Exception as e:
        print(f"Error calculating technical sentiment: {e}")
    
    return sentiment

def combine_sentiment_signals(technical_sentiment: float, 
                            news_sentiment: float, 
                            news_weight: float = 0.3) -> float:
    """
    Combine technical and news sentiment signals
    
    Args:
        technical_sentiment: Technical analysis sentiment score (-1 to 1)
        news_sentiment: News sentiment score (-1 to 1)
        news_weight: Weight to give news sentiment (0 to 1)
    
    Returns:
        Combined sentiment score (-1 to 1)
    """
    tech_weight = 1 - news_weight
    return (technical_sentiment * tech_weight) + (news_sentiment * news_weight)

def get_sentiment_recommendation(sentiment_score: float) -> str:
    """Convert sentiment score to trading recommendation"""
    if sentiment_score >= 0.6:
        return "STRONG_BUY"
    elif sentiment_score >= 0.2:
        return "BUY"
    elif sentiment_score <= -0.6:
        return "STRONG_SELL"
    elif sentiment_score <= -0.2:
        return "SELL"
    else:
        return "HOLD"

def analyze_stock_sentiment(df: pd.DataFrame, 
                          symbol: str, 
                          news_analyzer: Optional[NewsAnalyzer] = None,
                          news_days: int = 7,
                          news_weight: float = 0.3) -> Dict:
    """
    Perform comprehensive sentiment analysis for a stock
    
    Args:
        df: DataFrame with technical indicators
        symbol: Stock symbol
        news_analyzer: Optional NewsAnalyzer instance
        news_days: Number of days of news to analyze
        news_weight: Weight to give news sentiment
    
    Returns:
        Dictionary with sentiment analysis results
    """
    try:
        # Calculate technical sentiment
        tech_sentiment = calculate_flexible_technical_sentiment(df).iloc[-1]
        
        # Get news sentiment if analyzer is provided
        news_sentiment = 0.0
        news_count = 0
        if news_analyzer is not None:
            news_data = news_analyzer.get_news_sentiment(symbol, days=news_days)
            news_sentiment = news_data['sentiment_score']
            news_count = news_data['article_count']
        
        # Combine signals
        combined_sentiment = combine_sentiment_signals(
            tech_sentiment, news_sentiment, news_weight
        )
        
        # Generate recommendation
        recommendation = get_sentiment_recommendation(combined_sentiment)
        
        return {
            'symbol': symbol,
            'technical_sentiment': tech_sentiment,
            'news_sentiment': news_sentiment,
            'news_count': news_count,
            'combined_sentiment': combined_sentiment,
            'recommendation': recommendation
        }
        
    except Exception as e:
        print(f"Error analyzing sentiment for {symbol}: {e}")
        return {
            'symbol': symbol,
            'error': str(e),
            'recommendation': 'ERROR'
        }
