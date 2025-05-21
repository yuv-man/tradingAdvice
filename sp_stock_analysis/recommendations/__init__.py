"""
Stock trading recommendations package that combines ML predictions,
technical analysis, and sentiment analysis.
"""

from .engine import RecommendationEngine, get_recommendation_details

__all__ = [
    'RecommendationEngine',
    'get_recommendation_details'
] 