"""
Main script for stock analysis combining ML prediction, sentiment analysis,
technical analysis, and visualization.
"""

import os
import sys
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

from ..data.fetchers import get_historical_data, get_sp500_symbols
from ..analysis.technical import calculate_technical_indicators
from ..analysis.sentiment import analyze_stock_sentiment
from ..models.predictors import StockPredictor
from ..visualization.charts import plot_results, save_results_html
from ..data.news import NewsAnalyzer

class StockAnalyzer:
    def __init__(self):
        """Initialize the stock analyzer with all required components"""
        # Load environment variables
        load_dotenv()
        
        # Setup output directory
        self.output_dir = os.getenv('OUTPUT_DIR', './results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.predictor = StockPredictor()
        self.news_analyzer = NewsAnalyzer()
        
    def analyze_single_stock(self, symbol: str, show_details: bool = True) -> dict:
        """Analyze a single stock and return recommendations"""
        try:
            print(f"\nAnalyzing {symbol}...")
            
            # 1. Get historical data
            df = get_historical_data(symbol)
            if df is None or len(df) < 50:
                return {
                    'symbol': symbol,
                    'status': 'Failed',
                    'error': 'Insufficient historical data',
                    'recommendation': 'INSUFFICIENT_DATA'
                }
            
            # 2. Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # 3. ML Analysis
            X, y = self.predictor.prepare_features(df)
            model_scores = self.predictor.train_models(X, y)
            ml_probability = self.predictor.predict_probability(X.iloc[-1:])
            # Get individual model predictions
            model_predictions = self.predictor.get_model_predictions(X.iloc[-1:])
            
            # 4. Sentiment Analysis
            sentiment_results = analyze_stock_sentiment(
                df, 
                symbol, 
                news_analyzer=self.news_analyzer
            )
            
            # 5. Combine results
            latest_price = df['Close'].iloc[-1]
            latest_volume = df['Volume'].iloc[-1]
            
            result = {
                'symbol': symbol,
                'last_price': latest_price,
                'volume': latest_volume,
                'technical_sentiment': sentiment_results['technical_sentiment'],
                'news_sentiment': sentiment_results['news_sentiment'],
                'ml_probability': ml_probability,
                'ml_confidence': max(model_scores.values()) if model_scores else 0.5,
                'model_scores': model_scores,
                'model_predictions': model_predictions,
                'recommendation': sentiment_results['recommendation'],
                'rsi': df['RSI'].iloc[-1],
                'macd': df['MACD'].iloc[-1],
                'status': 'Success'
            }
            
            if show_details:
                self._print_analysis_details(result)
            
            return result
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'status': 'Failed',
                'error': str(e),
                'recommendation': 'ERROR'
            }
    
    def analyze_multiple_stocks(self, symbols: list, max_stocks: int = None) -> pd.DataFrame:
        """Analyze multiple stocks and return results as DataFrame"""
        if max_stocks:
            symbols = symbols[:max_stocks]
        
        results = []
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Analyzing {symbol}")
            result = self.analyze_single_stock(symbol, show_details=False)
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame([r for r in results if r['status'] == 'Success'])
        
        if len(df) > 0:
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.output_dir, f"stock_analysis_{timestamp}.csv")
            df.to_csv(output_file, index=False)
            
            # Create visualizations
            if len(df) >= 3:
                plot_results(df)
                save_results_html(df, self.output_dir)
        
        return df
    
    def _print_analysis_details(self, result: dict):
        """Print detailed analysis results"""
        print("\n=== Analysis Results ===")
        print(f"Symbol: {result['symbol']}")
        print(f"Last Price: ${result['last_price']:.2f}")
        print(f"Recommendation: {result['recommendation']}")
        
        print("\nML Analysis:")
        print(f"Overall ML Probability: {result['ml_probability']:.2f}")
        print(f"ML Confidence: {result['ml_confidence']:.2f}")
        print("\nIndividual Model Results:")
        for model_name, score in result['model_scores'].items():
            prediction = result['model_predictions'][model_name]
            print(f"{model_name}:")
            print(f"  - Score: {score:.3f}")
            print(f"  - Prediction: {prediction:.3f}")
        
        print("\nIndicators:")
        print(f"RSI: {result['rsi']:.2f}")
        print(f"MACD: {result['macd']:.2f}")
        print(f"Technical Sentiment: {result['technical_sentiment']:.2f}")
        print(f"News Sentiment: {result['news_sentiment']:.2f}")

def main():
    """Main execution function"""
    analyzer = StockAnalyzer()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        # Single stock analysis
        symbol = sys.argv[1].upper()
        analyzer.analyze_single_stock(symbol)
    else:
        # Multiple stocks analysis
        max_stocks = int(os.getenv('MAX_SYMBOLS', '10'))
        symbols = get_sp500_symbols()
        print(f"Analyzing {max_stocks} stocks from S&P 500...")
        analyzer.analyze_multiple_stocks(symbols, max_stocks=max_stocks)

if __name__ == "__main__":
    main()
