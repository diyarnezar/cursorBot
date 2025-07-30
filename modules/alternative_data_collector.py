import requests
import pandas as pd
import numpy as np
import logging
import time
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf
from textblob import TextBlob
import re
import os

# Free data sources
try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.techindicators import TechIndicators
    from alpha_vantage.fundamentaldata import FundamentalData
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    logging.warning("Alpha Vantage not available. Macro indicators will be limited.")

class AlternativeDataCollector:
    """
    Comprehensive alternative data collector using free sources
    for maximum intelligence and market edge.
    """
    
    def __init__(self, 
                 alpha_vantage_key: str = None,
                 cache_dir: str = "data/alternative",
                 update_interval: int = 300):
        """
        Initialize the alternative data collector.
        
        Args:
            alpha_vantage_key: Free API key from Alpha Vantage
            cache_dir: Directory to cache data
            update_interval: Update interval in seconds
        """
        self.alpha_vantage_key = alpha_vantage_key
        self.cache_dir = cache_dir
        self.update_interval = update_interval
        self.last_update = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize data sources
        if alpha_vantage_key and ALPHA_VANTAGE_AVAILABLE:
            self.ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
            self.ti = TechIndicators(key=alpha_vantage_key, output_format='pandas')
            self.fd = FundamentalData(key=alpha_vantage_key, output_format='pandas')
        else:
            self.ts = None
            self.ti = None
            self.fd = None
        
        logging.info("🚀 Alternative Data Collector initialized with free sources")
    
    def get_onchain_metrics(self) -> Dict[str, float]:
        """
        Collect on-chain metrics using free APIs.
        
        Returns:
            Dictionary of on-chain metrics
        """
        try:
            metrics = {}
            
            # Ethereum network metrics (free APIs)
            eth_metrics = self._get_ethereum_metrics()
            metrics.update(eth_metrics)
            
            # DeFi metrics
            defi_metrics = self._get_defi_metrics()
            metrics.update(defi_metrics)
            
            # Gas metrics
            gas_metrics = self._get_gas_metrics()
            metrics.update(gas_metrics)
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error collecting on-chain metrics: {e}")
            return {}
    
    def _get_ethereum_metrics(self) -> Dict[str, float]:
        """Get Ethereum network metrics from free APIs."""
        try:
            # Use free Ethereum APIs
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': 'ethereum',
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            eth_data = data.get('ethereum', {})
            
            return {
                'eth_market_cap': eth_data.get('usd_market_cap', 0),
                'eth_24h_volume': eth_data.get('usd_24h_vol', 0),
                'eth_24h_change': eth_data.get('usd_24h_change', 0),
                'eth_price': eth_data.get('usd', 0)
            }
            
        except Exception as e:
            logging.error(f"Error getting Ethereum metrics: {e}")
            return {}
    
    def _get_defi_metrics(self) -> Dict[str, float]:
        """Get DeFi metrics from free APIs."""
        try:
            # DeFi Pulse Index (free)
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': 'defi-pulse-index',
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            defi_data = data.get('defi-pulse-index', {})
            
            return {
                'defi_pulse_index': defi_data.get('usd', 0),
                'defi_24h_change': defi_data.get('usd_24h_change', 0)
            }
            
        except Exception as e:
            logging.error(f"Error getting DeFi metrics: {e}")
            return {}
    
    def _get_gas_metrics(self) -> Dict[str, float]:
        """Get gas metrics from free APIs."""
        try:
            # Ethereum gas tracker (free)
            url = "https://api.etherscan.io/api"
            params = {
                'module': 'gastracker',
                'action': 'gasoracle',
                'apikey': 'YourApiKeyToken'  # Free tier
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data['status'] == '1':
                result = data['result']
                return {
                    'gas_safe_low': int(result.get('SafeLow', 0)),
                    'gas_standard': int(result.get('ProposeGasPrice', 0)),
                    'gas_fast': int(result.get('FastGasPrice', 0)),
                    'gas_rapid': int(result.get('suggestBaseFee', 0))
                }
            
            return {}
            
        except Exception as e:
            logging.error(f"Error getting gas metrics: {e}")
            return {}
    
    def get_social_sentiment(self) -> Dict[str, float]:
        """
        Collect social sentiment from free sources.
        
        Returns:
            Dictionary of sentiment metrics
        """
        try:
            sentiment_data = {}
            
            # Reddit sentiment
            reddit_sentiment = self._get_reddit_sentiment()
            sentiment_data.update(reddit_sentiment)
            
            # Twitter sentiment (using free APIs)
            twitter_sentiment = self._get_twitter_sentiment()
            sentiment_data.update(twitter_sentiment)
            
            # Crypto news sentiment
            news_sentiment = self._get_news_sentiment()
            sentiment_data.update(news_sentiment)
            
            return sentiment_data
            
        except Exception as e:
            logging.error(f"Error collecting social sentiment: {e}")
            return {}
    
    def _get_reddit_sentiment(self) -> Dict[str, float]:
        """Get Reddit sentiment from r/cryptocurrency and r/ethereum."""
        try:
            # Use Reddit's free API
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            subreddits = ['cryptocurrency', 'ethereum', 'defi']
            sentiment_scores = []
            
            for subreddit in subreddits:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
                response = requests.get(url, headers=headers, timeout=10)
                data = response.json()
                
                posts = data['data']['children']
                for post in posts:
                    title = post['data']['title']
                    score = post['data']['score']
                    
                    # Simple sentiment analysis
                    blob = TextBlob(title)
                    sentiment = blob.sentiment.polarity
                    
                    # Weight by post score
                    weighted_sentiment = sentiment * (score / 1000)
                    sentiment_scores.append(weighted_sentiment)
            
            if sentiment_scores:
                return {
                    'reddit_sentiment': np.mean(sentiment_scores),
                    'reddit_sentiment_std': np.std(sentiment_scores),
                    'reddit_engagement': len(sentiment_scores)
                }
            
            return {}
            
        except Exception as e:
            logging.error(f"Error getting Reddit sentiment: {e}")
            return {}
    
    def _get_twitter_sentiment(self) -> Dict[str, float]:
        """Get Twitter sentiment using free sentiment APIs."""
        try:
            # Use free sentiment analysis APIs
            # Note: For production, consider using Twitter API v2 with free tier
            
            # Simulate Twitter sentiment for now
            # In production, integrate with Twitter API or sentiment services
            
            return {
                'twitter_sentiment': np.random.normal(0, 0.1),  # Placeholder
                'twitter_volume': np.random.randint(1000, 10000)  # Placeholder
            }
            
        except Exception as e:
            logging.error(f"Error getting Twitter sentiment: {e}")
            return {}
    
    def _get_news_sentiment(self) -> Dict[str, float]:
        """Get crypto news sentiment from free news APIs."""
        try:
            # Use free news APIs
            url = "https://cryptonews-api.com/api/v1/news"
            params = {
                'tickers': 'ETH',
                'items': 50,
                'token': 'demo'  # Free demo token
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'data' in data:
                news_items = data['data']
                sentiments = []
                
                for item in news_items:
                    title = item.get('title', '')
                    text = item.get('text', '')
                    
                    # Analyze sentiment
                    blob = TextBlob(title + ' ' + text)
                    sentiment = blob.sentiment.polarity
                    sentiments.append(sentiment)
                
                if sentiments:
                    return {
                        'news_sentiment': np.mean(sentiments),
                        'news_sentiment_std': np.std(sentiments),
                        'news_count': len(sentiments)
                    }
            
            return {}
            
        except Exception as e:
            logging.error(f"Error getting news sentiment: {e}")
            return {}
    
    def get_macro_indicators(self) -> Dict[str, float]:
        """
        Collect macroeconomic indicators using free sources.
        
        Returns:
            Dictionary of macro indicators
        """
        try:
            macro_data = {}
            
            # US Dollar Index (DXY)
            dxy_data = self._get_dxy_data()
            macro_data.update(dxy_data)
            
            # VIX (Fear Index)
            vix_data = self._get_vix_data()
            macro_data.update(vix_data)
            
            # Gold price
            gold_data = self._get_gold_data()
            macro_data.update(gold_data)
            
            # Treasury yields
            treasury_data = self._get_treasury_data()
            macro_data.update(treasury_data)
            
            return macro_data
            
        except Exception as e:
            logging.error(f"Error collecting macro indicators: {e}")
            return {}
    
    def _get_dxy_data(self) -> Dict[str, float]:
        """Get US Dollar Index data."""
        try:
            # Use yfinance for free data
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="5d")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'dxy_price': latest['Close'],
                    'dxy_change': (latest['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100
                }
            
            return {}
            
        except Exception as e:
            logging.error(f"Error getting DXY data: {e}")
            return {}
    
    def _get_vix_data(self) -> Dict[str, float]:
        """Get VIX (Fear Index) data."""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'vix_price': latest['Close'],
                    'vix_change': (latest['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100
                }
            
            return {}
            
        except Exception as e:
            logging.error(f"Error getting VIX data: {e}")
            return {}
    
    def _get_gold_data(self) -> Dict[str, float]:
        """Get gold price data."""
        try:
            gold = yf.Ticker("GC=F")
            hist = gold.history(period="5d")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'gold_price': latest['Close'],
                    'gold_change': (latest['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100
                }
            
            return {}
            
        except Exception as e:
            logging.error(f"Error getting gold data: {e}")
            return {}
    
    def _get_treasury_data(self) -> Dict[str, float]:
        """Get Treasury yield data."""
        try:
            # 10-year Treasury yield
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="5d")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'treasury_10y': latest['Close'],
                    'treasury_10y_change': (latest['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100
                }
            
            return {}
            
        except Exception as e:
            logging.error(f"Error getting Treasury data: {e}")
            return {}
    
    def get_all_alternative_data(self) -> Dict[str, Any]:
        """
        Collect all alternative data sources.
        
        Returns:
            Comprehensive dictionary of all alternative data
        """
        try:
            all_data = {}
            
            # On-chain metrics
            onchain_data = self.get_onchain_metrics()
            all_data.update(onchain_data)
            
            # Social sentiment
            sentiment_data = self.get_social_sentiment()
            all_data.update(sentiment_data)
            
            # Macro indicators
            macro_data = self.get_macro_indicators()
            all_data.update(macro_data)
            
            # Add timestamp
            all_data['timestamp'] = datetime.now().isoformat()
            
            # Cache the data
            self._cache_data(all_data)
            
            return all_data
            
        except Exception as e:
            logging.error(f"Error collecting all alternative data: {e}")
            return {}
    
    def _cache_data(self, data: Dict[str, Any]) -> None:
        """Cache the collected data."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.cache_dir}/alternative_data_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Keep only last 100 files
            files = sorted([f for f in os.listdir(self.cache_dir) if f.startswith('alternative_data_')])
            if len(files) > 100:
                for old_file in files[:-100]:
                    os.remove(os.path.join(self.cache_dir, old_file))
                    
        except Exception as e:
            logging.error(f"Error caching data: {e}")
    
    def get_cached_data(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Get cached alternative data from the last N hours.
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            List of cached data dictionaries
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            cached_data = []
            
            for filename in os.listdir(self.cache_dir):
                if filename.startswith('alternative_data_'):
                    filepath = os.path.join(self.cache_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    
                    if file_time >= cutoff_time:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            cached_data.append(data)
            
            return sorted(cached_data, key=lambda x: x.get('timestamp', ''))
            
        except Exception as e:
            logging.error(f"Error getting cached data: {e}")
            return []
    
    def create_alternative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create alternative data features from collected data.
        
        Args:
            df: Original DataFrame with OHLCV data
            
        Returns:
            DataFrame with alternative features added
        """
        try:
            # Get recent alternative data
            alt_data = self.get_cached_data(hours_back=24)
            
            if not alt_data:
                return df
            
            # Create features from alternative data
            features = {}
            
            # On-chain features
            if alt_data:
                latest = alt_data[-1]
                
                # Gas features
                features['gas_safe_low'] = latest.get('gas_safe_low', 0)
                features['gas_standard'] = latest.get('gas_standard', 0)
                features['gas_fast'] = latest.get('gas_fast', 0)
                
                # Market features
                features['eth_market_cap'] = latest.get('eth_market_cap', 0)
                features['eth_24h_volume'] = latest.get('eth_24h_volume', 0)
                features['eth_24h_change'] = latest.get('eth_24h_change', 0)
                
                # Sentiment features
                features['reddit_sentiment'] = latest.get('reddit_sentiment', 0)
                features['news_sentiment'] = latest.get('news_sentiment', 0)
                
                # Macro features
                features['dxy_price'] = latest.get('dxy_price', 0)
                features['vix_price'] = latest.get('vix_price', 0)
                features['gold_price'] = latest.get('gold_price', 0)
                features['treasury_10y'] = latest.get('treasury_10y', 0)
            
            # Add features to DataFrame
            for feature_name, feature_value in features.items():
                df[feature_name] = feature_value
            
            return df
            
        except Exception as e:
            logging.error(f"Error creating alternative features: {e}")
            return df 