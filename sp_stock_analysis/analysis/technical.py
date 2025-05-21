"""
Technical analysis module for stock analysis.
Provides functions for calculating various technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional

def ensure_series(data: Union[pd.DataFrame, pd.Series], column_name: Optional[str] = None) -> pd.Series:
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

def calculate_rsi(prices: Union[pd.DataFrame, pd.Series], period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)"""
    prices = ensure_series(prices)
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: Union[pd.DataFrame, pd.Series], 
                  fast: int = 12, 
                  slow: int = 26, 
                  signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    prices = ensure_series(prices)
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_atr(high: Union[pd.DataFrame, pd.Series], 
                 low: Union[pd.DataFrame, pd.Series], 
                 close: Union[pd.DataFrame, pd.Series], 
                 period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    high = ensure_series(high)
    low = ensure_series(low)
    close = ensure_series(close)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_bollinger_bands(prices: Union[pd.DataFrame, pd.Series], 
                            period: int = 20, 
                            std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    prices = ensure_series(prices)
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    width = (upper - lower) / middle
    return upper, middle, lower, width

def calculate_obv(close: Union[pd.DataFrame, pd.Series], 
                 volume: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    """Calculate On-Balance Volume (OBV)"""
    close = ensure_series(close)
    volume = ensure_series(volume)
    obv = pd.Series(0.0, index=close.index)
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    return obv

def calculate_stochastic(high: Union[pd.DataFrame, pd.Series], 
                        low: Union[pd.DataFrame, pd.Series], 
                        close: Union[pd.DataFrame, pd.Series], 
                        k_period: int = 14, 
                        d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator"""
    high = ensure_series(high)
    low = ensure_series(low)
    close = ensure_series(close)
    
    high_roll = high.rolling(window=k_period).max()
    low_roll = low.rolling(window=k_period).min()
    stoch_k = 100 * (close - low_roll) / (high_roll - low_roll)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calculate_adx(high: Union[pd.DataFrame, pd.Series], 
                 low: Union[pd.DataFrame, pd.Series], 
                 close: Union[pd.DataFrame, pd.Series], 
                 period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX)"""
    high = ensure_series(high)
    low = ensure_series(low)
    close = ensure_series(close)
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (abs(minus_dm.rolling(window=period).mean()) / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators for a given DataFrame"""
    try:
        # Basic price and volume metrics
        df['Daily_Return'] = df['Close'].pct_change()
        df['Daily_Volatility'] = df['Daily_Return'].rolling(window=20).std()  # Add volatility
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Moving averages
        df['SMA5'] = df['Close'].rolling(window=5).mean()
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA100'] = df['Close'].rolling(window=100).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Rate of Change (ROC)
        df['ROC5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['ROC10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        df['ROC20'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100
        
        # RSI
        df['RSI'] = calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['Signal_Line'], df['MACD_Hist'] = calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['BB_Width'] = calculate_bollinger_bands(df['Close'])
        
        # ATR
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df['High'], df['Low'], df['Close'])
        
        # ADX
        df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'])
        
        # OBV
        df['OBV'] = calculate_obv(df['Close'], df['Volume'])
        
        return df
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return df
