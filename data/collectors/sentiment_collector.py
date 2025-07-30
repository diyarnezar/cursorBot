"""
Sentiment Analysis Collector for Market Sentiment
Part of Project Hyperion - Ultimate Autonomous Trading Bot

Collects sentiment from:
- Reddit (cryptocurrency subreddits)
- Twitter (crypto-related tweets)
- News sentiment analysis
- Social media platforms
- Fear & Greed Index
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
import re
# from textblob import TextBlob  # Uncomment when textblob is installed
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class SentimentConfig:
    """Configuration for sentiment collection"""
    # API keys (free tiers)
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    twitter_bearer_token: str = ""
    
    # Rate limiting
    rate_limit_per_second: int = 2
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    
    # Collection parameters
    collection_interval: int = 600  # 10 minutes
    sentiment_window_hours: int = 24
    
    # Subreddits to monitor
    crypto_subreddits: Optional[List[str]] = None
    
    # Keywords for sentiment analysis
    positive_keywords: Optional[List[str]] = None
    negative_keywords: Optional[List[str]] = None


class SentimentCollector:
    """
    Advanced Sentiment Analysis Collector
    
    Features:
    - Multi-platform sentiment collection
    - Real-time sentiment analysis
    - Fear & Greed Index calculation
    - Sentiment trend analysis
    - Keyword-based sentiment scoring
    """
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.session = requests.Session()
        self.sentiment_data = {}
        self.sentiment_history = []
        self.keyword_scores = {}
        
        # Initialize default values
        if self.config.crypto_subreddits is None:
            self.config.crypto_subreddits = [
                'cryptocurrency', 'ethereum', 'bitcoin', 'defi', 'cryptomarkets'
            ]
        
        if self.config.positive_keywords is None:
            self.config.positive_keywords = [
                'bull', 'bullish', 'moon', 'pump', 'surge', 'rally', 'gain', 'up', 'positive',
                'buy', 'long', 'hodl', 'diamond', 'rocket', 'lambo', 'mooning', 'breakout'
            ]
        
        if self.config.negative_keywords is None:
            self.config.negative_keywords = [
                'bear', 'bearish', 'dump', 'crash', 'drop', 'fall', 'down', 'negative',
                'sell', 'short', 'fud', 'panic', 'dump', 'rekt', 'rekt', 'dumpster'
            ]
        
        logger.info("Sentiment Collector initialized")

    async def collect_all_sentiment(self) -> Dict[str, Any]:
        """Collect sentiment from all sources"""
        try:
            sentiment_data = {
                'reddit_sentiment': await self.collect_reddit_sentiment(),
                'twitter_sentiment': await self.collect_twitter_sentiment(),
                'news_sentiment': await self.collect_news_sentiment(),
                'fear_greed_index': await self.calculate_fear_greed_index(),
                'overall_sentiment': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate overall sentiment
            sentiment_data['overall_sentiment'] = self._calculate_overall_sentiment(sentiment_data)
            
            # Store sentiment data
            self._store_sentiment_data(sentiment_data)
            
            logger.info(f"Collected sentiment data from {len(sentiment_data) - 2} sources")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting sentiment data: {e}")
            return {'timestamp': datetime.now().isoformat()}

    async def collect_reddit_sentiment(self) -> Dict[str, Any]:
        """Collect Reddit sentiment from cryptocurrency subreddits"""
        try:
            reddit_sentiment = {
                'subreddit_sentiments': {},
                'overall_sentiment': 0.0,
                'total_posts': 0,
                'total_comments': 0,
                'engagement_score': 0.0
            }
            
            total_sentiment = 0.0
            total_posts = 0
            
            for subreddit in self.config.crypto_subreddits:
                subreddit_data = await self._collect_subreddit_sentiment(subreddit)
                reddit_sentiment['subreddit_sentiments'][subreddit] = subreddit_data
                
                total_sentiment += subreddit_data.get('sentiment_score', 0.0)
                total_posts += subreddit_data.get('post_count', 0)
            
            # Calculate overall Reddit sentiment
            if total_posts > 0:
                reddit_sentiment['overall_sentiment'] = total_sentiment / len(self.config.crypto_subreddits)
                reddit_sentiment['total_posts'] = total_posts
                reddit_sentiment['engagement_score'] = self._calculate_engagement_score(reddit_sentiment)
            
            return reddit_sentiment
            
        except Exception as e:
            logger.error(f"Error collecting Reddit sentiment: {e}")
            return {'overall_sentiment': 0.0, 'total_posts': 0}

    async def _collect_subreddit_sentiment(self, subreddit: str) -> Dict[str, Any]:
        """Collect sentiment from a specific subreddit"""
        try:
            # Simulate Reddit API call (in production, use Reddit API)
            # For now, generate realistic sentiment data
            
            # Simulate post data
            post_count = np.random.randint(50, 200)
            comment_count = np.random.randint(500, 2000)
            
            # Generate sentiment scores for posts
            post_sentiments = np.random.normal(0.0, 0.2, post_count)
            
            # Calculate subreddit sentiment
            sentiment_score = np.mean(post_sentiments)
            sentiment_std = np.std(post_sentiments)
            
            # Calculate engagement metrics
            upvote_ratio = np.random.uniform(0.7, 0.95)
            engagement_rate = (comment_count / post_count) if post_count > 0 else 0
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_std': sentiment_std,
                'post_count': post_count,
                'comment_count': comment_count,
                'upvote_ratio': upvote_ratio,
                'engagement_rate': engagement_rate,
                'top_keywords': self._extract_top_keywords(subreddit)
            }
            
        except Exception as e:
            logger.error(f"Error collecting subreddit sentiment for {subreddit}: {e}")
            return {'sentiment_score': 0.0, 'post_count': 0}

    async def collect_twitter_sentiment(self) -> Dict[str, Any]:
        """Collect Twitter sentiment for cryptocurrency"""
        try:
            twitter_sentiment = {
                'overall_sentiment': 0.0,
                'tweet_volume': 0,
                'hashtag_sentiments': {},
                'user_sentiments': {},
                'trending_topics': []
            }
            
            # Simulate Twitter data collection
            # In production, use Twitter API v2
            
            # Generate tweet volume
            tweet_volume = np.random.randint(10000, 50000)
            
            # Generate sentiment scores
            tweet_sentiments = np.random.normal(0.0, 0.15, min(tweet_volume, 1000))
            overall_sentiment = np.mean(tweet_sentiments)
            
            # Generate hashtag sentiments
            hashtags = ['#crypto', '#ethereum', '#bitcoin', '#defi', '#altcoin']
            hashtag_sentiments = {}
            
            for hashtag in hashtags:
                hashtag_sentiments[hashtag] = {
                    'sentiment': np.random.normal(0.0, 0.2),
                    'volume': np.random.randint(1000, 10000)
                }
            
            twitter_sentiment.update({
                'overall_sentiment': overall_sentiment,
                'tweet_volume': tweet_volume,
                'hashtag_sentiments': hashtag_sentiments,
                'trending_topics': ['crypto', 'ethereum', 'defi', 'nft']
            })
            
            return twitter_sentiment
            
        except Exception as e:
            logger.error(f"Error collecting Twitter sentiment: {e}")
            return {'overall_sentiment': 0.0, 'tweet_volume': 0}

    async def collect_news_sentiment(self) -> Dict[str, Any]:
        """Collect news sentiment for cryptocurrency"""
        try:
            news_sentiment = {
                'overall_sentiment': 0.0,
                'article_count': 0,
                'source_sentiments': {},
                'topic_sentiments': {},
                'sentiment_trend': 'neutral'
            }
            
            # Simulate news sentiment collection
            # In production, use news APIs and NLP analysis
            
            # Generate article data
            article_count = np.random.randint(20, 100)
            article_sentiments = np.random.normal(0.0, 0.25, article_count)
            
            # Calculate overall sentiment
            overall_sentiment = np.mean(article_sentiments)
            
            # Generate source sentiments
            sources = ['coindesk', 'cointelegraph', 'decrypt', 'theblock']
            source_sentiments = {}
            
            for source in sources:
                source_sentiments[source] = {
                    'sentiment': np.random.normal(0.0, 0.3),
                    'article_count': np.random.randint(5, 25)
                }
            
            # Generate topic sentiments
            topics = ['defi', 'nft', 'regulation', 'adoption', 'technology']
            topic_sentiments = {}
            
            for topic in topics:
                topic_sentiments[topic] = np.random.normal(0.0, 0.2)
            
            # Determine sentiment trend
            if overall_sentiment > 0.1:
                sentiment_trend = 'positive'
            elif overall_sentiment < -0.1:
                sentiment_trend = 'negative'
            else:
                sentiment_trend = 'neutral'
            
            news_sentiment.update({
                'overall_sentiment': overall_sentiment,
                'article_count': article_count,
                'source_sentiments': source_sentiments,
                'topic_sentiments': topic_sentiments,
                'sentiment_trend': sentiment_trend
            })
            
            return news_sentiment
            
        except Exception as e:
            logger.error(f"Error collecting news sentiment: {e}")
            return {'overall_sentiment': 0.0, 'article_count': 0}

    async def calculate_fear_greed_index(self) -> Dict[str, Any]:
        """Calculate Fear & Greed Index for cryptocurrency"""
        try:
            # Get recent sentiment data
            recent_sentiment = self._get_recent_sentiment_data()
            
            # Calculate Fear & Greed components
            volatility = self._calculate_volatility_component()
            market_momentum = self._calculate_momentum_component()
            social_volume = self._calculate_social_volume_component()
            dominance = self._calculate_dominance_component()
            google_trends = self._calculate_google_trends_component()
            
            # Calculate Fear & Greed Index (0-100)
            fear_greed_score = (
                volatility * 0.25 +
                market_momentum * 0.25 +
                social_volume * 0.15 +
                dominance * 0.10 +
                google_trends * 0.25
            )
            
            # Determine sentiment category
            if fear_greed_score >= 75:
                sentiment = 'Extreme Greed'
            elif fear_greed_score >= 60:
                sentiment = 'Greed'
            elif fear_greed_score >= 40:
                sentiment = 'Neutral'
            elif fear_greed_score >= 25:
                sentiment = 'Fear'
            else:
                sentiment = 'Extreme Fear'
            
            return {
                'fear_greed_score': fear_greed_score,
                'sentiment': sentiment,
                'components': {
                    'volatility': volatility,
                    'market_momentum': market_momentum,
                    'social_volume': social_volume,
                    'dominance': dominance,
                    'google_trends': google_trends
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating Fear & Greed Index: {e}")
            return {'fear_greed_score': 50, 'sentiment': 'Neutral'}

    def _calculate_volatility_component(self) -> float:
        """Calculate volatility component for Fear & Greed Index"""
        try:
            # Simulate volatility calculation
            # In production, calculate actual price volatility
            volatility = np.random.uniform(0, 100)
            return volatility
        except Exception as e:
            logger.error(f"Error calculating volatility component: {e}")
            return 50.0

    def _calculate_momentum_component(self) -> float:
        """Calculate market momentum component"""
        try:
            # Simulate momentum calculation
            momentum = np.random.uniform(0, 100)
            return momentum
        except Exception as e:
            logger.error(f"Error calculating momentum component: {e}")
            return 50.0

    def _calculate_social_volume_component(self) -> float:
        """Calculate social volume component"""
        try:
            # Simulate social volume calculation
            social_volume = np.random.uniform(0, 100)
            return social_volume
        except Exception as e:
            logger.error(f"Error calculating social volume component: {e}")
            return 50.0

    def _calculate_dominance_component(self) -> float:
        """Calculate dominance component"""
        try:
            # Simulate dominance calculation
            dominance = np.random.uniform(0, 100)
            return dominance
        except Exception as e:
            logger.error(f"Error calculating dominance component: {e}")
            return 50.0

    def _calculate_google_trends_component(self) -> float:
        """Calculate Google Trends component"""
        try:
            # Simulate Google Trends calculation
            google_trends = np.random.uniform(0, 100)
            return google_trends
        except Exception as e:
            logger.error(f"Error calculating Google Trends component: {e}")
            return 50.0

    def _calculate_overall_sentiment(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate overall sentiment from all sources"""
        try:
            # Weighted average of different sentiment sources
            reddit_sentiment = sentiment_data.get('reddit_sentiment', {}).get('overall_sentiment', 0.0)
            twitter_sentiment = sentiment_data.get('twitter_sentiment', {}).get('overall_sentiment', 0.0)
            news_sentiment = sentiment_data.get('news_sentiment', {}).get('overall_sentiment', 0.0)
            fear_greed_score = sentiment_data.get('fear_greed_index', {}).get('fear_greed_score', 50.0)
            
            # Normalize Fear & Greed score to -1 to 1 range
            fear_greed_normalized = (fear_greed_score - 50) / 50
            
            # Weighted average (can be adjusted based on importance)
            overall_sentiment = (
                reddit_sentiment * 0.3 +
                twitter_sentiment * 0.3 +
                news_sentiment * 0.2 +
                fear_greed_normalized * 0.2
            )
            
            return overall_sentiment
            
        except Exception as e:
            logger.error(f"Error calculating overall sentiment: {e}")
            return 0.0

    def _calculate_engagement_score(self, reddit_data: Dict[str, Any]) -> float:
        """Calculate engagement score for Reddit data"""
        try:
            total_posts = reddit_data.get('total_posts', 0)
            total_comments = sum(
                subreddit.get('comment_count', 0) 
                for subreddit in reddit_data.get('subreddit_sentiments', {}).values()
            )
            
            if total_posts > 0:
                engagement_score = total_comments / total_posts
                return min(engagement_score / 10, 1.0)  # Normalize to 0-1
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.0

    def _extract_top_keywords(self, subreddit: str) -> List[str]:
        """Extract top keywords from subreddit (simulated)"""
        try:
            # Simulate keyword extraction
            # In production, use NLP to extract actual keywords
            keywords = ['ethereum', 'defi', 'nft', 'crypto', 'bitcoin']
            return np.random.choice(keywords, size=3, replace=False).tolist()
        except Exception as e:
            logger.error(f"Error extracting keywords for {subreddit}: {e}")
            return []

    def _get_recent_sentiment_data(self) -> List[Dict[str, Any]]:
        """Get recent sentiment data for calculations"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config.sentiment_window_hours)
            recent_data = [
                data for data in self.sentiment_history
                if datetime.fromisoformat(data.get('timestamp', '')) >= cutoff_time
            ]
            return recent_data
        except Exception as e:
            logger.error(f"Error getting recent sentiment data: {e}")
            return []

    def _store_sentiment_data(self, sentiment_data: Dict[str, Any]):
        """Store sentiment data"""
        try:
            # Store in memory
            self.sentiment_history.append(sentiment_data)
            
            # Keep only recent data (last 7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            self.sentiment_history = [
                data for data in self.sentiment_history
                if datetime.fromisoformat(data.get('timestamp', '')) >= cutoff_time
            ]
            
            # Store to file
            timestamp = datetime.now()
            filename = f"sentiment_data_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = f"data/sentiment/{filename}"
            
            with open(filepath, 'w') as f:
                json.dump(sentiment_data, f, indent=2)
            
            logger.info(f"Stored sentiment data: {filepath}")
            
        except Exception as e:
            logger.error(f"Error storing sentiment data: {e}")

    def get_sentiment_trend(self, hours: int = 24) -> Dict[str, Any]:
        """Get sentiment trend over specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = [
                data for data in self.sentiment_history
                if datetime.fromisoformat(data.get('timestamp', '')) >= cutoff_time
            ]
            
            if not recent_data:
                return {'trend': 'neutral', 'change': 0.0}
            
            # Calculate trend
            sentiments = [data.get('overall_sentiment', 0.0) for data in recent_data]
            if len(sentiments) >= 2:
                trend_change = sentiments[-1] - sentiments[0]
                
                if trend_change > 0.1:
                    trend = 'increasing'
                elif trend_change < -0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                return {
                    'trend': trend,
                    'change': trend_change,
                    'current_sentiment': sentiments[-1],
                    'average_sentiment': np.mean(sentiments)
                }
            
            return {'trend': 'neutral', 'change': 0.0}
            
        except Exception as e:
            logger.error(f"Error getting sentiment trend: {e}")
            return {'trend': 'neutral', 'change': 0.0}

    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get sentiment summary statistics"""
        try:
            if not self.sentiment_history:
                return {'total_records': 0}
            
            sentiments = [data.get('overall_sentiment', 0.0) for data in self.sentiment_history]
            
            return {
                'total_records': len(self.sentiment_history),
                'average_sentiment': np.mean(sentiments),
                'sentiment_std': np.std(sentiments),
                'min_sentiment': np.min(sentiments),
                'max_sentiment': np.max(sentiments),
                'last_updated': self.sentiment_history[-1].get('timestamp', '')
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {'total_records': 0}


# Example usage
if __name__ == "__main__":
    config = SentimentConfig()
    collector = SentimentCollector(config)
    
    # Run sentiment collection
    asyncio.run(collector.collect_all_sentiment()) 