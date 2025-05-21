import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

load_dotenv()
# Configure Gemini API
def setup_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')

# Get S&P 500 symbols
def get_sp500_symbols():
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
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']

# Fetch historical data
def get_historical_data(symbol, days=360):  # Increased to 360 days for better ML training
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None

# Calculate comprehensive technical features
def calculate_technical_features(data):
    if data is None or len(data) < 100:
        return None
    
    df = data.copy()
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        if period >= 10:
            df[f'Close_SMA{period}_Ratio'] = df['Close'] / df[f'SMA{period}']
    
    # Price Rate of Change
    for period in [5, 10, 20, 50]:
        df[f'ROC{period}'] = df['Close'].pct_change(periods=period) * 100
    
    # Momentum indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal_Line'], df['MACD_Histogram'] = calculate_macd(df['Close'])
    
    # Volatility indicators
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['BB_Width'] = calculate_bollinger_bands(df['Close'])
    
    # Volume indicators
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']
    df['OBV'] = calculate_obv(df['Close'], df['Volume'])
    
    # Stochastic Oscillator
    df['K_percent'], df['D_percent'] = calculate_stochastic(df['High'], df['Low'], df['Close'])
    
    # Advanced features
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std() * np.sqrt(252)
    df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252)
    
    # Trend strength
    df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'])
    
    # Create target variables
    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = ((df['Next_Close'] > df['Close']) * 1).astype(int)
    df['Target_Return'] = ((df['Next_Close'] - df['Close']) / df['Close']) * 100
    
    # Shift some features to create lagged variables
    for feature in ['Close', 'Volume', 'RSI', 'MACD']:
        df[f'{feature}_Lag1'] = df[feature].shift(1)
        df[f'{feature}_Lag2'] = df[feature].shift(2)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

# Technical indicator calculation functions
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    width = (upper - lower) / middle
    return upper, middle, lower, width

def calculate_obv(close, volume):
    obv = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    return obv

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    high_roll = high.rolling(window=k_period).max()
    low_roll = low.rolling(window=k_period).min()
    stoch_k = 100 * (close - low_roll) / (high_roll - low_roll)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calculate_adx(high, low, close, period=14):
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

# Get sentiment analysis
def get_sentiment_analysis(symbol, model):
    prompt = f"""
    Analyze the current market sentiment for {symbol} stock. 
    Consider recent news, analyst opinions, and market trends.
    Rate the sentiment on a scale from -1.0 (extremely negative) to 1.0 (extremely positive).
    Provide only the numerical score without any additional text.
    """
    
    try:
        response = model.generate_content(prompt)
        sentiment_text = response.text.strip()
        
        try:
            sentiment_score = float(sentiment_text)
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
        except ValueError:
            print(f"Could not parse sentiment for {symbol}: {sentiment_text}")
            sentiment_score = 0.0
            
        return sentiment_score
    except Exception as e:
        print(f"Error getting sentiment for {symbol}: {e}")
        return 0.0

# Create ML models
def create_ensemble_models():
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    return models

# Train and predict with ML models
def train_and_predict(data, sentiment_score):
    if data is None or len(data) < 100:
        return None
    
    # Add sentiment to features
    data['Sentiment'] = sentiment_score
    
    # Select features for ML
    feature_cols = [col for col in data.columns if col not in ['Target', 'Target_Return', 'Next_Close']]
    
    # Remove non-numeric columns
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]
    
    X = data[feature_cols]
    y = data['Target']
    
    # Split data with time series consideration
    train_size = int(len(data) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = create_ensemble_models()
    model_results = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            probabilities = model.predict_proba(X_test_scaled)[:, 1]
            accuracy = accuracy_score(y_test, predictions)
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'prediction': predictions[-1],
                'probability': probabilities[-1]
            }
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    # Ensemble prediction (voting)
    predictions = []
    probabilities = []
    
    for result in model_results.values():
        predictions.append(result['prediction'])
        probabilities.append(result['probability'])
    
    if predictions:
        ensemble_prediction = np.mean(predictions)
        ensemble_probability = np.mean(probabilities)
        average_accuracy = np.mean([r['accuracy'] for r in model_results.values()])
        
        return {
            'prediction': ensemble_prediction,
            'probability': ensemble_probability,
            'accuracy': average_accuracy,
            'model_results': model_results,
            'feature_importance': get_feature_importance(models, feature_cols)
        }
    
    return None

def get_feature_importance(models, feature_cols):
    importance_dict = {}
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance_dict[name] = dict(zip(feature_cols, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            importance_dict[name] = dict(zip(feature_cols, np.abs(model.coef_[0])))
    
    return importance_dict

# Generate comprehensive recommendation
def generate_enhanced_recommendation(technical_data, sentiment_score, ml_result):
    if technical_data is None or len(technical_data) < 5:
        return "INSUFFICIENT_DATA"
    
    latest = technical_data.iloc[-1]
    
    # Technical analysis score
    technical_score = 0
    
    # Moving Average signals
    ma_signals = 0
    if latest['Close'] > latest['SMA20']: ma_signals += 1
    if latest['Close'] > latest['SMA50']: ma_signals += 1
    if latest['Close'] > latest['SMA200']: ma_signals += 2
    if latest['SMA20'] > latest['SMA50']: ma_signals += 1
    if latest['SMA50'] > latest['SMA200']: ma_signals += 1
    technical_score += ma_signals - 3  # Normalize to -3 to 3
    
    # Momentum signals
    if latest['RSI'] < 30: technical_score += 2
    elif latest['RSI'] > 70: technical_score -= 2
    
    if latest['MACD'] > latest['Signal_Line']: technical_score += 1
    else: technical_score -= 1
    
    # Trend strength
    if latest['ADX'] > 50: 
        if latest['Close'] > latest.get('SMA50', latest['Close']): 
            technical_score += 1
        else: 
            technical_score -= 1
    
    # Normalize technical score
    technical_score = max(-8, min(8, technical_score))
    normalized_technical = technical_score / 8.0
    
    # ML prediction weight
    ml_weight = 0
    ml_confidence = 0.5
    
    if ml_result:
        ml_weight = (ml_result['probability'] - 0.5) * 2  # Scale to -1 to 1
        ml_confidence = ml_result['accuracy']
    
    # Combine all scores (weights: 40% technical, 40% ML, 20% sentiment)
    final_score = (normalized_technical * 0.4) + (ml_weight * 0.4) + (sentiment_score * 0.2)
    
    # Adjust recommendation based on ML confidence
    if ml_confidence < 0.6:
        # Low confidence, make more conservative recommendations
        if final_score > 0.5:
            return "WEAK_BUY"
        elif final_score > 0:
            return "HOLD"
        elif final_score > -0.5:
            return "HOLD"
        else:
            return "WEAK_SELL"
    else:
        # High confidence recommendations
        if final_score > 0.4:
            return "STRONG_BUY"
        elif final_score > 0.15:
            return "BUY"
        elif final_score > -0.15:
            return "HOLD"
        elif final_score > -0.4:
            return "SELL"
        else:
            return "STRONG_SELL"

# Main analysis function
def analyze_sp500_with_ml():
    symbols = get_sp500_symbols()
    model = setup_gemini()
    results = []
    
    print(f"Analyzing {len(symbols)} symbols with ML...")
    
    for i, symbol in enumerate(symbols[:20]):  # Limit to first 20 for demo
        print(f"Analyzing {symbol} ({i+1}/{min(len(symbols), 20)})")
        
        try:
            # Get historical data
            historical_data = get_historical_data(symbol)
            
            # Calculate technical features
            technical_data = calculate_technical_features(historical_data)
            
            # Get sentiment score
            sentiment_score = get_sentiment_analysis(symbol, model)
            
            # Train and predict with ML
            ml_result = train_and_predict(technical_data, sentiment_score)
            
            # Generate enhanced recommendation
            recommendation = generate_enhanced_recommendation(
                technical_data, 
                sentiment_score, 
                ml_result
            )
            
            # Create result entry
            result = {
                'Symbol': symbol,
                'Last_Price': None if technical_data is None else technical_data['Close'].iloc[-1],
                'RSI': None if technical_data is None else technical_data['RSI'].iloc[-1],
                'SMA20': None if technical_data is None else technical_data['SMA20'].iloc[-1],
                'SMA50': None if technical_data is None else technical_data['SMA50'].iloc[-1],
                'SMA200': None if technical_data is None else technical_data['SMA200'].iloc[-1],
                'MACD': None if technical_data is None else technical_data['MACD'].iloc[-1],
                'ADX': None if technical_data is None else technical_data['ADX'].iloc[-1],
                'Sentiment': sentiment_score,
                'Recommendation': recommendation
            }
            
            # Add ML results if available
            if ml_result:
                result['ML_Prediction'] = ml_result['prediction']
                result['ML_Probability'] = ml_result['probability']
                result['ML_Accuracy'] = ml_result['accuracy']
            
            results.append(result)
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue
        
        # API rate limiting
        if (i + 1) % 5 == 0:
            print("Pausing for rate limiting...")
            time.sleep(10)
    
    return pd.DataFrame(results)

# Run the analysis
if __name__ == "__main__":
    print("Starting Enhanced S&P 500 Analysis with ML...")
    results_df = analyze_sp500_with_ml()
    
    # Save results to CSV
    output_file = f"sp500_ml_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"Analysis complete. Results saved to {output_file}")
    
    # Display recommendations
    print("\nRecommendation Summary:")
    print(results_df['Recommendation'].value_counts())
    
    # Show top buy recommendations
    buy_stocks = results_df[results_df['Recommendation'].isin(['STRONG_BUY', 'BUY'])].sort_values('ML_Probability', ascending=False)
    print("\nTOP BUY RECOMMENDATIONS:")
    print(buy_stocks[['Symbol', 'Last_Price', 'RSI', 'ADX', 'Sentiment', 'ML_Probability', 'ML_Accuracy', 'Recommendation']])
    
    # Show top sell recommendations
    sell_stocks = results_df[results_df['Recommendation'].isin(['STRONG_SELL', 'SELL'])].sort_values('ML_Probability', ascending=True)
    print("\nTOP SELL RECOMMENDATIONS:")
    print(sell_stocks[['Symbol', 'Last_Price', 'RSI', 'ADX', 'Sentiment', 'ML_Probability', 'ML_Accuracy', 'Recommendation']])
    
    # Save detailed analysis for top 5 recommendations
    top_recommendations = pd.concat([buy_stocks.head(5), sell_stocks.head(5)])
    top_recommendations.to_html(f"top_recommendations_{datetime.now().strftime('%Y%m%d')}.html", index=False)
    print(f"\nDetailed HTML report saved for top recommendations")