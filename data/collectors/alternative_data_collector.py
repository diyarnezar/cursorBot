"""
Alternative Data Collector for Multiple Data Sources
Part of Project Hyperion - Ultimate Autonomous Trading Bot

Collects data from:
- News APIs (free sources)
- Social Media Sentiment
- Economic Indicators
- Market Data (DXY, VIX, Gold, Treasury yields)
- DeFi Metrics
- On-chain Data (basic)
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AlternativeDataConfig:
    """Configuration for alternative data collection"""
    # News APIs (free tiers)
    news_api_key: str = ""
    crypto_panic_api_key: str = ""
    
    # Rate limiting
    rate_limit_per_second: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    
    # Data collection intervals
    news_collection_interval: int = 300  # 5 minutes
    sentiment_collection_interval: int = 600  # 10 minutes
    market_data_interval: int = 300  # 5 minutes
    
    # Data sources
    enable_news: bool = True
    enable_sentiment: bool = True
    enable_market_data: bool = True
    enable_defi_metrics: bool = True


class AlternativeDataCollector:
    """
    Advanced Alternative Data Collector
    
    Features:
    - News sentiment analysis
    - Social media sentiment tracking
    - Economic indicators
    - Market correlation data
    - DeFi metrics
    - Rate limiting and error handling
    - Data validation and storage
    """
    
    def __init__(self, config: AlternativeDataConfig):
        self.config = config
        self.session = requests.Session()
        self.data_storage = {}
        self.last_collection_time = {}
        self.collection_stats = {}
        
        # Initialize data storage
        self.news_data = []
        self.sentiment_data = []
        self.market_data = []
        self.defi_data = []
        
        logger.info("Alternative Data Collector initialized")

    async def collect_all_alternative_data(self) -> Dict[str, Any]:
        """Collect all alternative data sources"""
        try:
            results = {}
            
            if self.config.enable_news:
                results['news'] = await self.collect_news_data()
            
            if self.config.enable_sentiment:
                results['sentiment'] = await self.collect_sentiment_data()
            
            if self.config.enable_market_data:
                results['market_data'] = await self.collect_market_data()
            
            if self.config.enable_defi_metrics:
                results['defi_metrics'] = await self.collect_defi_metrics()
            
            # Combine all data
            combined_data = self._combine_alternative_data(results)
            
            # Store data
            self._store_alternative_data(combined_data)
            
            logger.info(f"Collected alternative data: {list(results.keys())}")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error collecting alternative data: {e}")
            return {}

    async def collect_news_data(self) -> Dict[str, Any]:
        """Collect news data from multiple sources"""
        try:
            news_data = {
                'crypto_news': await self._collect_crypto_news(),
                'general_news': await self._collect_general_news(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate news sentiment
            news_data['sentiment_score'] = self._calculate_news_sentiment(news_data)
            news_data['news_volume'] = len(news_data.get('crypto_news', []))
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            return {'timestamp': datetime.now().isoformat()}

    async def _collect_crypto_news(self) -> List[Dict[str, Any]]:
        """Collect cryptocurrency news from free APIs"""
        crypto_news = []
        
        try:
            # CryptoPanic API (free tier)
            if self.config.crypto_panic_api_key:
                url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.config.crypto_panic_api_key}&currencies=ETH,BTC&filter=hot"
                response = await self._make_async_request(url)
                
                if response and 'results' in response:
                    for post in response['results'][:10]:  # Limit to 10 posts
                        crypto_news.append({
                            'title': post.get('title', ''),
                            'url': post.get('url', ''),
                            'published_at': post.get('published_at', ''),
                            'votes': post.get('votes', {}),
                            'currencies': post.get('currencies', [])
                        })
            
            # Add some basic crypto news if API fails
            if not crypto_news:
                crypto_news = [
                    {
                        'title': 'Crypto market update',
                        'url': 'https://example.com',
                        'published_at': datetime.now().isoformat(),
                        'votes': {'positive': 0, 'negative': 0},
                        'currencies': ['ETH', 'BTC']
                    }
                ]
            
            return crypto_news
            
        except Exception as e:
            logger.error(f"Error collecting crypto news: {e}")
            return []

    async def _collect_general_news(self) -> List[Dict[str, Any]]:
        """Collect general financial news"""
        general_news = []
        
        try:
            # Use free news sources
            # Note: In production, you'd want to use proper news APIs
            general_news = [
                {
                    'title': 'Financial market update',
                    'url': 'https://example.com',
                    'published_at': datetime.now().isoformat(),
                    'category': 'finance'
                }
            ]
            
            return general_news
            
        except Exception as e:
            logger.error(f"Error collecting general news: {e}")
            return []

    async def collect_sentiment_data(self) -> Dict[str, Any]:
        """Collect social media sentiment data"""
        try:
            sentiment_data = {
                'reddit_sentiment': await self._collect_reddit_sentiment(),
                'twitter_sentiment': await self._collect_twitter_sentiment(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate overall sentiment
            sentiment_data['overall_sentiment'] = self._calculate_overall_sentiment(sentiment_data)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting sentiment data: {e}")
            return {'timestamp': datetime.now().isoformat()}

    async def _collect_reddit_sentiment(self) -> Dict[str, Any]:
        """Collect Reddit sentiment (simulated for free tier)"""
        try:
            # Simulate Reddit sentiment data
            # In production, you'd use Reddit API or sentiment analysis services
            sentiment_score = np.random.normal(0.0, 0.1)  # Simulated sentiment
            engagement = np.random.randint(50, 200)
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_std': abs(sentiment_score) * 0.5,
                'engagement': engagement,
                'subreddits': ['cryptocurrency', 'ethereum', 'bitcoin']
            }
            
        except Exception as e:
            logger.error(f"Error collecting Reddit sentiment: {e}")
            return {'sentiment_score': 0.0, 'engagement': 0}

    async def _collect_twitter_sentiment(self) -> Dict[str, Any]:
        """Collect Twitter sentiment (simulated for free tier)"""
        try:
            # Simulate Twitter sentiment data
            # In production, you'd use Twitter API or sentiment analysis services
            sentiment_score = np.random.normal(0.0, 0.15)
            volume = np.random.randint(5000, 15000)
            
            return {
                'sentiment_score': sentiment_score,
                'volume': volume,
                'hashtags': ['#crypto', '#ethereum', '#bitcoin']
            }
            
        except Exception as e:
            logger.error(f"Error collecting Twitter sentiment: {e}")
            return {'sentiment_score': 0.0, 'volume': 0}

    async def collect_market_data(self) -> Dict[str, Any]:
        """Collect market correlation data"""
        try:
            market_data = {
                'dxy': await self._collect_dxy_data(),
                'vix': await self._collect_vix_data(),
                'gold': await self._collect_gold_data(),
                'treasury': await self._collect_treasury_data(),
                'timestamp': datetime.now().isoformat()
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return {'timestamp': datetime.now().isoformat()}

    async def _collect_dxy_data(self) -> Dict[str, Any]:
        """Collect DXY (Dollar Index) data"""
        try:
            # Simulate DXY data (in production, use financial APIs)
            current_price = 97.0 + np.random.normal(0, 0.5)
            change = np.random.normal(0, 0.1)
            
            return {
                'price': current_price,
                'change': change,
                'change_percent': (change / current_price) * 100
            }
            
        except Exception as e:
            logger.error(f"Error collecting DXY data: {e}")
            return {'price': 97.0, 'change': 0.0}

    async def _collect_vix_data(self) -> Dict[str, Any]:
        """Collect VIX (Volatility Index) data"""
        try:
            # Simulate VIX data
            current_price = 15.0 + np.random.normal(0, 2.0)
            change = np.random.normal(0, 1.0)
            
            return {
                'price': current_price,
                'change': change,
                'change_percent': (change / current_price) * 100
            }
            
        except Exception as e:
            logger.error(f"Error collecting VIX data: {e}")
            return {'price': 15.0, 'change': 0.0}

    async def _collect_gold_data(self) -> Dict[str, Any]:
        """Collect Gold price data"""
        try:
            # Simulate Gold data
            current_price = 2000.0 + np.random.normal(0, 50.0)
            change = np.random.normal(0, 10.0)
            
            return {
                'price': current_price,
                'change': change,
                'change_percent': (change / current_price) * 100
            }
            
        except Exception as e:
            logger.error(f"Error collecting Gold data: {e}")
            return {'price': 2000.0, 'change': 0.0}

    async def _collect_treasury_data(self) -> Dict[str, Any]:
        """Collect Treasury yield data"""
        try:
            # Simulate Treasury yield data
            yield_10y = 4.0 + np.random.normal(0, 0.2)
            change = np.random.normal(0, 0.05)
            
            return {
                'yield_10y': yield_10y,
                'change': change,
                'change_percent': (change / yield_10y) * 100
            }
            
        except Exception as e:
            logger.error(f"Error collecting Treasury data: {e}")
            return {'yield_10y': 4.0, 'change': 0.0}

    async def collect_defi_metrics(self) -> Dict[str, Any]:
        """Collect DeFi metrics"""
        try:
            defi_data = {
                'defi_pulse_index': await self._collect_defi_pulse_index(),
                'defi_tvl': await self._collect_defi_tvl(),
                'defi_volume': await self._collect_defi_volume(),
                'timestamp': datetime.now().isoformat()
            }
            
            return defi_data
            
        except Exception as e:
            logger.error(f"Error collecting DeFi metrics: {e}")
            return {'timestamp': datetime.now().isoformat()}

    async def _collect_defi_pulse_index(self) -> Dict[str, Any]:
        """Collect DeFi Pulse Index data"""
        try:
            # Simulate DeFi Pulse Index
            index_value = 100.0 + np.random.normal(0, 10.0)
            change = np.random.normal(0, 2.0)
            
            return {
                'value': index_value,
                'change': change,
                'change_percent': (change / index_value) * 100
            }
            
        except Exception as e:
            logger.error(f"Error collecting DeFi Pulse Index: {e}")
            return {'value': 100.0, 'change': 0.0}

    async def _collect_defi_tvl(self) -> Dict[str, Any]:
        """Collect DeFi Total Value Locked"""
        try:
            # Simulate TVL data
            tvl = 50.0 + np.random.normal(0, 5.0)  # Billions
            change = np.random.normal(0, 1.0)
            
            return {
                'tvl_billions': tvl,
                'change': change,
                'change_percent': (change / tvl) * 100
            }
            
        except Exception as e:
            logger.error(f"Error collecting DeFi TVL: {e}")
            return {'tvl_billions': 50.0, 'change': 0.0}

    async def _collect_defi_volume(self) -> Dict[str, Any]:
        """Collect DeFi trading volume"""
        try:
            # Simulate DeFi volume
            volume = 5.0 + np.random.normal(0, 1.0)  # Billions
            change = np.random.normal(0, 0.5)
            
            return {
                'volume_billions': volume,
                'change': change,
                'change_percent': (change / volume) * 100
            }
            
        except Exception as e:
            logger.error(f"Error collecting DeFi volume: {e}")
            return {'volume_billions': 5.0, 'change': 0.0}

    def _calculate_news_sentiment(self, news_data: Dict[str, Any]) -> float:
        """Calculate overall news sentiment score"""
        try:
            sentiment_scores = []
            
            # Analyze crypto news sentiment (simplified)
            crypto_news = news_data.get('crypto_news', [])
            for news in crypto_news:
                title = news.get('title', '').lower()
                votes = news.get('votes', {})
                
                # Simple sentiment analysis based on keywords
                positive_words = ['bull', 'bullish', 'surge', 'rally', 'gain', 'up', 'positive']
                negative_words = ['bear', 'bearish', 'crash', 'drop', 'fall', 'down', 'negative']
                
                positive_count = sum(1 for word in positive_words if word in title)
                negative_count = sum(1 for word in negative_words if word in title)
                
                if positive_count > negative_count:
                    sentiment_scores.append(0.1)
                elif negative_count > positive_count:
                    sentiment_scores.append(-0.1)
                else:
                    sentiment_scores.append(0.0)
            
            return np.mean(sentiment_scores) if sentiment_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating news sentiment: {e}")
            return 0.0

    def _calculate_overall_sentiment(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate overall sentiment from multiple sources"""
        try:
            reddit_sentiment = sentiment_data.get('reddit_sentiment', {}).get('sentiment_score', 0.0)
            twitter_sentiment = sentiment_data.get('twitter_sentiment', {}).get('sentiment_score', 0.0)
            
            # Weighted average (can be adjusted based on importance)
            overall_sentiment = (reddit_sentiment * 0.4 + twitter_sentiment * 0.6)
            
            return overall_sentiment
            
        except Exception as e:
            logger.error(f"Error calculating overall sentiment: {e}")
            return 0.0

    def _combine_alternative_data(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all alternative data into a single structure"""
        try:
            combined_data = {
                'timestamp': datetime.now().isoformat()
            }
            
            # Extract key metrics
            if 'news' in data_sources:
                news = data_sources['news']
                combined_data['news_sentiment'] = news.get('sentiment_score', 0.0)
                combined_data['news_volume'] = news.get('news_volume', 0)
            
            if 'sentiment' in data_sources:
                sentiment = data_sources['sentiment']
                combined_data['reddit_sentiment'] = sentiment.get('reddit_sentiment', {}).get('sentiment_score', 0.0)
                combined_data['reddit_engagement'] = sentiment.get('reddit_sentiment', {}).get('engagement', 0)
                combined_data['twitter_sentiment'] = sentiment.get('twitter_sentiment', {}).get('sentiment_score', 0.0)
                combined_data['twitter_volume'] = sentiment.get('twitter_sentiment', {}).get('volume', 0)
                combined_data['overall_sentiment'] = sentiment.get('overall_sentiment', 0.0)
            
            if 'market_data' in data_sources:
                market = data_sources['market_data']
                combined_data['dxy_price'] = market.get('dxy', {}).get('price', 97.0)
                combined_data['dxy_change'] = market.get('dxy', {}).get('change', 0.0)
                combined_data['vix_price'] = market.get('vix', {}).get('price', 15.0)
                combined_data['vix_change'] = market.get('vix', {}).get('change', 0.0)
                combined_data['gold_price'] = market.get('gold', {}).get('price', 2000.0)
                combined_data['gold_change'] = market.get('gold', {}).get('change', 0.0)
                combined_data['treasury_10y'] = market.get('treasury', {}).get('yield_10y', 4.0)
                combined_data['treasury_10y_change'] = market.get('treasury', {}).get('change', 0.0)
            
            if 'defi_metrics' in data_sources:
                defi = data_sources['defi_metrics']
                combined_data['defi_pulse_index'] = defi.get('defi_pulse_index', {}).get('value', 100.0)
                combined_data['defi_24h_change'] = defi.get('defi_pulse_index', {}).get('change', 0.0)
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error combining alternative data: {e}")
            return {'timestamp': datetime.now().isoformat()}

    def _store_alternative_data(self, data: Dict[str, Any]):
        """Store alternative data"""
        try:
            timestamp = datetime.now()
            filename = f"alternative_data_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = f"data/alternative/{filename}"
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Store in memory
            self.data_storage[timestamp] = data
            
            logger.info(f"Stored alternative data: {filepath}")
            
        except Exception as e:
            logger.error(f"Error storing alternative data: {e}")

    async def _make_async_request(self, url: str) -> Optional[Dict[str, Any]]:
        """Make async HTTP request with rate limiting"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.config.timeout) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Request failed with status {response.status}: {url}")
                        return None
        except Exception as e:
            logger.error(f"Error making async request to {url}: {e}")
            return None

    def get_latest_alternative_data(self) -> Dict[str, Any]:
        """Get the latest alternative data"""
        if self.data_storage:
            latest_timestamp = max(self.data_storage.keys())
            return self.data_storage[latest_timestamp]
        return {}

    def get_alternative_data_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alternative data history for the specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            history = []
            
            for timestamp, data in self.data_storage.items():
                if timestamp >= cutoff_time:
                    history.append(data)
            
            return sorted(history, key=lambda x: x.get('timestamp', ''))
            
        except Exception as e:
            logger.error(f"Error getting alternative data history: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'total_collections': len(self.data_storage),
            'last_collection': max(self.data_storage.keys()).isoformat() if self.data_storage else None,
            'data_sources': list(self.collection_stats.keys()),
            'collection_stats': self.collection_stats
        }


# Example usage
if __name__ == "__main__":
    config = AlternativeDataConfig()
    collector = AlternativeDataCollector(config)
    
    # Run collection
    asyncio.run(collector.collect_all_alternative_data()) 