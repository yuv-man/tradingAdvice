"""
Recommendation engine that combines ML predictions, technical analysis, and sentiment analysis
to generate trading recommendations.
"""

from typing import Dict, Optional
import pandas as pd
from ..models import StockPredictor
from ..analysis.sentiment import (
    calculate_flexible_technical_sentiment,
    combine_sentiment_signals,
    get_sentiment_recommendation
)
from ..data.news import NewsAnalyzer

class RecommendationEngine:
    def __init__(self):
        self.predictor = StockPredictor()
        self.news_analyzer = NewsAnalyzer()
        
    def analyze_stock(self, 
                     df: pd.DataFrame, 
                     symbol: str,
                     include_news: bool = True,
                     news_weight: float = 0.3) -> Dict:
        """
        Perform comprehensive stock analysis and generate recommendations
        
        Args:
            df: DataFrame with technical indicators
            symbol: Stock symbol
            include_news: Whether to include news sentiment analysis
            news_weight: Weight to give news sentiment (0 to 1)
            
        Returns:
            Dictionary containing analysis results and recommendations
        """
        try:
            print(f"Analyzing {symbol}...")
            
            # 1. ML Analysis
            X, y = self.predictor.prepare_features(df)
            if len(X) < 20:
                return {
                    'symbol': symbol,
                    'error': 'Insufficient data for analysis',
                    'recommendation': 'INSUFFICIENT_DATA'
                }
            
            # Train models and get prediction
            model_scores = self.predictor.train_models(X, y)
            ml_probability = self.predictor.predict_probability(X.iloc[-1:])
            
            print(f"  ML Analysis complete - Probability: {ml_probability:.3f}")
            
            # 2. Technical Sentiment
            tech_sentiment = calculate_flexible_technical_sentiment(df).iloc[-1]
            print(f"  Technical sentiment: {tech_sentiment:.3f}")
            
            # 3. News Sentiment (if enabled)
            news_sentiment = 0.0
            news_count = 0
            if include_news and self.news_analyzer:
                try:
                    news_data = self.news_analyzer.get_news_sentiment(symbol, days=7)
                    news_sentiment = news_data['sentiment_score']
                    news_count = news_data['article_count']
                    print(f"  News sentiment: {news_sentiment:.3f} ({news_count} articles)")
                except Exception as e:
                    print(f"  Warning: News analysis failed - {e}")
            
            # 4. Combine Signals
            # Scale ML probability to -1 to 1 range
            ml_signal = (ml_probability - 0.5) * 2
            
            # Get ML confidence from model scores
            ml_confidence = max(model_scores.values()) if model_scores else 0.5
            
            # Combine signals using the existing function
            signals = {
                'technical': tech_sentiment,
                'ml': ml_signal,
                'news': news_sentiment
            }
            
            weights = {
                'technical': 0.50 if ml_confidence < 0.65 else 0.35,
                'ml': 0.25 if ml_confidence < 0.65 else 0.45,
                'news': 0.25 if ml_confidence < 0.65 else 0.20
            }
            
            final_score = combine_sentiment_signals(signals, weights)
            print(f"  Final combined score: {final_score:.3f}")
            
            # Generate recommendation using the existing function
            recommendation = get_sentiment_recommendation(
                final_score,
                confidence=ml_confidence
            )
            
            # 6. Prepare Results
            latest_price = df['Close'].iloc[-1]
            latest_volume = df['Volume'].iloc[-1]
            
            return {
                'symbol': symbol,
                'last_price': latest_price,
                'volume': latest_volume,
                'technical_sentiment': tech_sentiment,
                'ml_probability': ml_probability,
                'ml_confidence': ml_confidence,
                'news_sentiment': news_sentiment,
                'news_count': news_count,
                'final_score': final_score,
                'recommendation': recommendation,
                'rsi': df['RSI'].iloc[-1] if 'RSI' in df else None,
                'macd': df['MACD'].iloc[-1] if 'MACD' in df else None,
                'status': 'Success'
            }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'status': 'Failed',
                'recommendation': 'ERROR'
            }

def get_recommendation_details(recommendation: Dict) -> str:
    """Generate a detailed explanation of the recommendation"""
    if recommendation.get('status') != 'Success':
        return f"Analysis failed: {recommendation.get('error', 'Unknown error')}"
    
    details = []
    details.append(f"Recommendation: {recommendation['recommendation']}")
    details.append(f"Last Price: ${recommendation['last_price']:.2f}")
    
    if recommendation.get('rsi') is not None:
        details.append(f"RSI: {recommendation['rsi']:.1f}")
    
    if recommendation.get('technical_sentiment') is not None:
        details.append(f"Technical Sentiment: {recommendation['technical_sentiment']:.3f}")
    
    if recommendation.get('ml_probability') is not None:
        details.append(f"ML Probability: {recommendation['ml_probability']:.3f}")
        details.append(f"ML Confidence: {recommendation['ml_confidence']:.3f}")
    
    if recommendation.get('news_sentiment') != 0:
        details.append(f"News Sentiment: {recommendation['news_sentiment']:.3f}")
        details.append(f"News Articles Analyzed: {recommendation['news_count']}")
    
    details.append(f"Final Score: {recommendation['final_score']:.3f}")
    
    return "\n".join(details)
