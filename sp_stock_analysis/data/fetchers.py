import os
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import random

def validate_data_structure(data, symbol):
    """Validate and clean the data structure"""
    if data is None or len(data) == 0:
        return None
    
    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten columns by taking the first level
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"Error: Missing columns {missing_columns} for {symbol}")
        return None
    
    # Convert to numeric and handle any potential issues
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Remove rows with any NaN values
    initial_rows = len(data)
    data = data.dropna()
    if len(data) < initial_rows:
        print(f"Removed {initial_rows - len(data)} rows with NaN values for {symbol}")
    
    # Check if we still have enough data
    if len(data) < 50:
        print(f"Error: Insufficient data for {symbol}. Only {len(data)} rows available.")
        return None
    
    return data

def get_sp500_symbols():
    """Fetch S&P 500 symbols from Wikipedia"""
    print("Fetching S&P 500 symbols...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        
        symbols = []
        for row in table.findAll('tr')[1:]:
            symbol = row.findAll('td')[0].text.strip()
            symbols.append(symbol)
        
        print(f"Found {len(symbols)} S&P 500 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        # Return a small fallback list of major companies
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']

def get_historical_data(symbol, days=360):
    """Download historical stock data using yfinance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        # Validate and clean the data structure
        data = validate_data_structure(data, symbol)
        
        return data
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None

def fetch_with_retry(symbol, max_retries=3, initial_delay=1):
    """Fetch data with retry logic and exponential backoff"""
    for attempt in range(max_retries):
        try:
            data = get_historical_data(symbol)
            if data is not None:
                return data
            
            delay = initial_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
            
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            
    print(f"Failed to fetch data for {symbol} after {max_retries} attempts")
    return None

def fetch_batch(symbols, batch_size=5, delay_range=(1, 3)):
    """Fetch data for multiple symbols in batches with random delays"""
    results = {}
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}")
        
        for symbol in batch:
            data = fetch_with_retry(symbol)
            if data is not None:
                results[symbol] = data
            
            # Random delay between requests
            if symbol != batch[-1]:  # No need to delay after the last symbol in batch
                delay = random.uniform(delay_range[0], delay_range[1])
                time.sleep(delay)
        
        # Longer delay between batches
        if i + batch_size < len(symbols):
            batch_delay = random.uniform(delay_range[0] * 2, delay_range[1] * 2)
            print(f"Batch complete. Waiting {batch_delay:.1f}s before next batch...")
            time.sleep(batch_delay)
    
    return results
