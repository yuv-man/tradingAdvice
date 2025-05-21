import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sp_stock_analysis.analysis.technical import (
    calculate_rsi, calculate_macd, calculate_atr,
    calculate_bollinger_bands, calculate_obv,
    calculate_stochastic, calculate_adx
)

def ensure_series(data, column_name=None):
    """Ensure data is a pandas Series"""
    if isinstance(data, pd.DataFrame):
        if column_name and column_name in data.columns:
            return data[column_name]
        else:
            return data.iloc[:, 0]  # Take first column
    elif isinstance(data, pd.Series):
        return data
    else:
        return pd.Series(data)

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for feature engineering"""
    features = df.copy()
    
    # Basic price features
    features['Daily_Return'] = features['Close'].pct_change()
    features['Log_Return'] = np.log(features['Close'] / features['Close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20, 50, 200]:
        features[f'SMA{period}'] = features['Close'].rolling(window=period).mean()
        features[f'EMA{period}'] = features['Close'].ewm(span=period, adjust=False).mean()
    
    # Volume features
    features['Volume_SMA20'] = features['Volume'].rolling(window=20).mean()
    features['Volume_Ratio'] = features['Volume'] / features['Volume_SMA20']
    
    # Volatility features
    features['Daily_Volatility'] = features['Daily_Return'].rolling(window=20).std()
    features['ATR'] = calculate_atr(features['High'], features['Low'], features['Close'])
    
    # Momentum indicators
    features['RSI'] = calculate_rsi(features['Close'])
    macd, signal, hist = calculate_macd(features['Close'])
    features['MACD'] = macd
    features['Signal_Line'] = signal
    features['MACD_Hist'] = hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(features['Close'])
    features['BB_Upper'] = bb_upper
    features['BB_Middle'] = bb_middle
    features['BB_Lower'] = bb_lower
    features['BB_Width'] = bb_width
    
    # Rate of Change
    features['ROC5'] = calculate_roc(features['Close'], 5)
    features['ROC10'] = calculate_roc(features['Close'], 10)
    features['ROC20'] = calculate_roc(features['Close'], 20)
    
    # Additional indicators
    features['OBV'] = calculate_obv(features['Close'], features['Volume'])
    stoch_k, stoch_d = calculate_stochastic(features['High'], features['Low'], features['Close'])
    features['Stoch_K'] = stoch_k
    features['Stoch_D'] = stoch_d
    features['ADX'] = calculate_adx(features['High'], features['Low'], features['Close'])
    
    return features

def calculate_roc(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Rate of Change"""
    return (prices - prices.shift(period)) / prices.shift(period) * 100
