import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import time
import random
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """Class for fetching and analyzing financial news"""
    
    def __init__(self, sentiment_model=None):
        """Initialize NewsAnalyzer with optional sentiment model"""
        self.sentiment_model = sentiment_model
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        
    def get_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Fetch news articles for a given stock symbol with enhanced robustness
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of news to fetch
            
        Returns:
            List of news articles with title, date, url, and source
        """
        try:
            articles = []
            
            # Try multiple news sources in order of reliability
            sources = [
                ('finviz', self._get_finviz_news),
                ('alpha_vantage', self._get_alpha_vantage_news),
                ('newsapi', self._get_newsapi_news),
                ('yahoo', self._get_yahoo_news),
                ('marketwatch', self._get_marketwatch_news)
            ]
            
            for source_name, source_func in sources:
                try:
                    logger.info(f"Trying {source_name} news source for {symbol}...")
                    new_articles = source_func(symbol, days)
                    if new_articles:
                        articles.extend(new_articles)
                        logger.info(f"Found {len(new_articles)} articles from {source_name}")
                        if len(articles) >= 15:  # Get more articles for better analysis
                            break
                    else:
                        logger.info(f"No articles found from {source_name}")
                except Exception as e:
                    logger.warning(f"Error fetching from {source_name}: {e}")
                    continue
            
            # Add small delay between sources
            time.sleep(0.5)
            
            # Deduplicate and sort articles
            seen_titles = set()
            unique_articles = []
            for article in articles:
                title = article['title'].lower()
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_articles.append(article)
            
            # Sort by date, newest first
            unique_articles.sort(key=lambda x: x['date'], reverse=True)
            
            # If no articles found, create synthetic entry
            if not unique_articles:
                logger.info(f"Using fallback synthetic news for {symbol}")
                unique_articles.append({
                    'title': f'{symbol} market activity',
                    'date': datetime.now(),
                    'url': '',
                    'source': 'synthetic',
                    'sentiment': 0.0
                })
            
            return unique_articles
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    def _get_alpha_vantage_news(self, symbol: str, days: int) -> List[Dict]:
        """Fetch news from Alpha Vantage API"""
        if not self.api_key:
            return []
            
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.api_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'feed' not in data:
                return []
                
            articles = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for item in data['feed']:
                pub_date = datetime.strptime(item['time_published'][:10], '%Y-%m-%d')
                if pub_date >= cutoff_date:
                    articles.append({
                        'title': item['title'],
                        'date': pub_date,
                        'url': item['url'],
                        'source': 'Alpha Vantage',
                        'sentiment': item.get('overall_sentiment_score', 0)
                    })
            
            return articles
            
        except Exception as e:
            logger.warning(f"Alpha Vantage news fetch failed: {e}")
            return []

    def _get_newsapi_news(self, symbol: str, days: int) -> List[Dict]:
        """Fetch news from NewsAPI"""
        if not self.news_api_key:
            return []
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={symbol} stock&"
            f"from={start_date.strftime('%Y-%m-%d')}&"
            f"to={end_date.strftime('%Y-%m-%d')}&"
            f"language=en&"
            f"sortBy=relevancy&"
            f"apiKey={self.news_api_key}"
        )
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if data.get('status') != 'ok':
                return []
                
            articles = []
            for item in data.get('articles', []):
                pub_date = datetime.strptime(item['publishedAt'][:10], '%Y-%m-%d')
                articles.append({
                    'title': item['title'],
                    'date': pub_date,
                    'url': item['url'],
                    'source': 'NewsAPI',
                    'sentiment': None
                })
            
            return articles
            
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
            return []

    def _get_finviz_news(self, symbol: str, days: int) -> List[Dict]:
        """Scrape news from FinViz"""
        url = f"https://finviz.com/quote.ashx?t={symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_table = soup.find(id='news-table')
            
            if not news_table:
                return []
                
            articles = []
            cutoff_date = datetime.now() - timedelta(days=days)
            current_date = None
            
            for row in news_table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 2:
                    date_str = cells[0].text.strip()
                    
                    # Handle different date formats
                    if 'Today' in date_str:
                        time_str = date_str.replace('Today ', '')
                        date = datetime.combine(
                            datetime.now().date(),
                            datetime.strptime(time_str, '%I:%M%p').time()
                        )
                    elif 'Yesterday' in date_str:
                        time_str = date_str.replace('Yesterday ', '')
                        date = datetime.combine(
                            (datetime.now() - timedelta(days=1)).date(),
                            datetime.strptime(time_str, '%I:%M%p').time()
                        )
                    else:
                        try:
                            if ' ' in date_str:  # Full date
                                date = datetime.strptime(date_str, '%b-%d-%y %I:%M%p')
                                current_date = date.date()
                            else:  # Time only
                                if current_date is None:
                                    current_date = datetime.now().date()
                                time = datetime.strptime(date_str, '%I:%M%p').time()
                                date = datetime.combine(current_date, time)
                        except ValueError as e:
                            logger.warning(f"Could not parse date: {date_str} - {e}")
                            continue
                    
                    if date.date() >= cutoff_date.date():
                        title = cells[1].a.text
                        url = cells[1].a['href']
                        articles.append({
                            'title': title,
                            'date': date,
                            'url': url,
                            'source': 'FinViz',
                            'sentiment': None
                        })
            
            return articles
            
        except Exception as e:
            logger.warning(f"FinViz scraping failed: {e}")
            return []

    def _get_yahoo_news(self, symbol: str, days: int) -> List[Dict]:
        """Scrape news from Yahoo Finance"""
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.find_all('div', {'class': 'Ov(h) Pend(44px) Pstart(25px)'})
            
            articles = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for item in news_items:
                try:
                    title_elem = item.find('h3')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    link = title_elem.find('a')['href']
                    if not link.startswith('http'):
                        link = 'https://finance.yahoo.com' + link
                    
                    # Yahoo typically shows recent news, assume current date if no date found
                    date = datetime.now()
                    
                    if date.date() >= cutoff_date.date():
                        articles.append({
                            'title': title,
                            'date': date,
                            'url': link,
                            'source': 'Yahoo Finance',
                            'sentiment': None
                        })
                except Exception as e:
                    logger.warning(f"Error parsing Yahoo news item: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.warning(f"Yahoo Finance scraping failed: {e}")
            return []

    def _get_marketwatch_news(self, symbol: str, days: int) -> List[Dict]:
        """Scrape news from MarketWatch"""
        url = f"https://www.marketwatch.com/investing/stock/{symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.find_all('div', {'class': 'article__content'})
            
            articles = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for item in news_items:
                try:
                    title_elem = item.find('a', {'class': 'link'})
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    link = title_elem['href']
                    if not link.startswith('http'):
                        link = 'https://www.marketwatch.com' + link
                    
                    # MarketWatch typically shows recent news, assume current date if no date found
                    date = datetime.now()
                    
                    if date.date() >= cutoff_date.date():
                        articles.append({
                            'title': title,
                            'date': date,
                            'url': link,
                            'source': 'MarketWatch',
                            'sentiment': None
                        })
                except Exception as e:
                    logger.warning(f"Error parsing MarketWatch news item: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.warning(f"MarketWatch scraping failed: {e}")
            return []

    def analyze_sentiment(self, articles: List[Dict]) -> Tuple[float, int]:
        """
        Enhanced sentiment analysis with multiple approaches
        
        Args:
            articles: List of news articles
            
        Returns:
            Tuple of (average sentiment score, number of articles analyzed)
        """
        if not articles:
            return 0.0, 0
            
        sentiments = []
        
        for article in articles:
            # Use pre-calculated sentiment if available
            if article.get('sentiment') is not None:
                sentiments.append(article['sentiment'])
                continue
                
            text = article['title']
            if article.get('summary'):
                text += " " + article['summary']
            
            # Use FinBERT model if available
            if self.sentiment_model:
                try:
                    result = self.sentiment_model(text[:512])  # Limit text length
                    if result and len(result) > 0:
                        sent_label = result[0]['label'].lower()
                        sent_score = result[0]['score']
                        
                        if 'positive' in sent_label:
                            sentiments.append(sent_score)
                        elif 'negative' in sent_label:
                            sentiments.append(-sent_score)
                        else:
                            sentiments.append(0.0)
                    else:
                        # Fallback to rule-based
                        score = self._enhanced_sentiment_analysis(text)
                        sentiments.append(score)
                except Exception as e:
                    logger.warning(f"FinBERT analysis failed: {e}")
                    score = self._enhanced_sentiment_analysis(text)
                    sentiments.append(score)
            else:
                # Use enhanced rule-based analysis
                score = self._enhanced_sentiment_analysis(text)
                sentiments.append(score)
        
        if not sentiments:
            return 0.0, 0
            
        # Weight recent articles more heavily
        weights = [1.0 - (i * 0.1) for i in range(len(sentiments))]
        weights = [w / sum(weights) for w in weights]
        
        weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights))
        return weighted_sentiment, len(sentiments)

    def _enhanced_sentiment_analysis(self, text: str) -> float:
        """Enhanced rule-based sentiment analysis"""
        text = text.lower()
        
        # Expanded financial word lists
        positive_words = {
            'surge': 1.5, 'jump': 1.3, 'gain': 1.0, 'rise': 1.0, 'climb': 1.0,
            'positive': 1.0, 'profit': 1.2, 'growth': 1.1, 'strong': 1.2,
            'upgrade': 1.4, 'beat': 1.3, 'exceed': 1.3, 'momentum': 1.1,
            'outperform': 1.3, 'opportunity': 1.0, 'promising': 1.1
        }
        
        negative_words = {
            'fall': -1.0, 'drop': -1.0, 'decline': -1.0, 'slip': -0.8,
            'negative': -1.0, 'loss': -1.2, 'weak': -1.0, 'risk': -0.9,
            'downgrade': -1.4, 'miss': -1.2, 'fail': -1.3, 'concern': -0.8,
            'underperform': -1.3, 'warning': -1.1, 'volatile': -0.7
        }
        
        # Intensity modifiers
        intensifiers = {
            'very': 1.5, 'significantly': 1.7, 'sharply': 1.8, 'dramatically': 2.0,
            'slightly': 0.7, 'somewhat': 0.8, 'marginally': 0.6
        }
        
        words = text.split()
        sentiment_score = 0
        total_matches = 0
        
        for i, word in enumerate(words):
            # Check for intensifiers
            intensity = 1.0
            if i > 0:
                prev_word = words[i-1].strip('.,!?;:"()[]')
                if prev_word in intensifiers:
                    intensity = intensifiers[prev_word]
            
            # Calculate sentiment with intensity
            if word in positive_words:
                sentiment_score += positive_words[word] * intensity
                total_matches += 1
            elif word in negative_words:
                sentiment_score += negative_words[word] * intensity
                total_matches += 1
        
        if total_matches == 0:
            return 0.0
            
        # Normalize to [-1, 1] range
        final_score = sentiment_score / (total_matches * 2)
        return max(-1.0, min(1.0, final_score))

    def get_news_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """
        Get overall news sentiment for a stock
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of news to analyze
            
        Returns:
            Dictionary with sentiment score and metadata
        """
        articles = self.get_news(symbol, days)
        sentiment_score, article_count = self.analyze_sentiment(articles)
        
        return {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'article_count': article_count,
            'period_days': days,
            'latest_articles': articles[:5]  # Return 5 most recent articles
        }
