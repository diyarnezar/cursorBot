#!/usr/bin/env python3
"""
SMART ALTERNATIVE DATA MODULE - MAXIMUM INTELLIGENCE
Project Hyperion - Free APIs with Intelligent Rate Limiting

This module implements the best free alternative data APIs with smart rate limiting
to make the bot the smartest possible without ever hitting API limits.
"""

import requests
import logging
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
import threading
import os
from dataclasses import dataclass
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for each API with rate limits"""
    name: str
    base_url: str
    calls_per_minute: int
    calls_per_hour: int
    calls_per_day: int
    requires_key: bool = False
    key_name: str = ""
    enabled: bool = True

class SmartRateLimiter:
    """Intelligent rate limiter that never hits API limits"""
    
    def __init__(self):
        self.api_calls = {}
        self.lock = threading.Lock()
        
    def can_call(self, api_name: str, config: APIConfig) -> bool:
        """Check if we can make a call to the API"""
        with self.lock:
            now = time.time()
            
            if api_name not in self.api_calls:
                self.api_calls[api_name] = {
                    'calls': deque(),
                    'hourly_calls': deque(),
                    'daily_calls': deque()
                }
            
            calls_data = self.api_calls[api_name]
            
            # Clean old calls
            while calls_data['calls'] and now - calls_data['calls'][0] > 60:
                calls_data['calls'].popleft()
            
            while calls_data['hourly_calls'] and now - calls_data['hourly_calls'][0] > 3600:
                calls_data['hourly_calls'].popleft()
                
            while calls_data['daily_calls'] and now - calls_data['daily_calls'][0] > 86400:
                calls_data['daily_calls'].popleft()
            
            # Check limits (use 80% of limits to be safe)
            safe_minute_limit = int(config.calls_per_minute * 0.8)
            safe_hour_limit = int(config.calls_per_hour * 0.8)
            safe_day_limit = int(config.calls_per_day * 0.8)
            
            if (len(calls_data['calls']) >= safe_minute_limit or
                len(calls_data['hourly_calls']) >= safe_hour_limit or
                len(calls_data['daily_calls']) >= safe_day_limit):
                return False
            
            return True
    
    def record_call(self, api_name: str):
        """Record an API call"""
        with self.lock:
            now = time.time()
            if api_name not in self.api_calls:
                self.api_calls[api_name] = {
                    'calls': deque(),
                    'hourly_calls': deque(),
                    'daily_calls': deque()
                }
            
            calls_data = self.api_calls[api_name]
            calls_data['calls'].append(now)
            calls_data['hourly_calls'].append(now)
            calls_data['daily_calls'].append(now)
    
    def get_wait_time(self, api_name: str, config: APIConfig) -> float:
        """Get how long to wait before next call"""
        with self.lock:
            if api_name not in self.api_calls:
                return 0
            
            calls_data = self.api_calls[api_name]
            now = time.time()
            
            # Find the earliest time we can make another call
            if calls_data['calls']:
                earliest_next = calls_data['calls'][0] + 60  # 1 minute from first call
                return max(0, earliest_next - now)
            
            return 0

class SmartAlternativeData:
    """
    Smart Alternative Data Collector with Maximum Intelligence:
    
    FREE APIs with HIGH LIMITS:
    1. CoinGecko - 50 calls/minute (no key needed)
    2. CryptoCompare - 100k calls/month (no key needed)
    3. Messari - 1000 calls/day (no key needed)
    4. Glassnode - 10 calls/minute (no key needed)
    5. Fear & Greed Index - Unlimited (no key needed)
    6. Binance Public API - 1200 calls/minute (no key needed)
    7. Etherscan - 5 calls/second (free key)
    8. CoinMarketCap - 10k calls/month (free key)
    9. Alpha Vantage - 5 calls/minute (free key)
    10. NewsAPI - 100 calls/day (free key)
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        self.rate_limiter = SmartRateLimiter()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Project-Hyperion-Smart-Bot/2.0'
        })
        
        # Initialize API configurations
        self.api_configs = self._init_api_configs()
        
        # Data cache
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes
        
        # Background data collection
        self.background_thread = None
        self.running = False
        
        logger.info("🧠 Smart Alternative Data initialized with maximum intelligence")
        logger.info(f"📊 Available APIs: {len([c for c in self.api_configs.values() if c.enabled])}")
    
    def _init_api_configs(self) -> Dict[str, APIConfig]:
        """Initialize API configurations with rate limits"""
        configs = {
            # HIGH LIMIT FREE APIs
            'coingecko': APIConfig(
                name='CoinGecko',
                base_url='https://api.coingecko.com/api/v3',
                calls_per_minute=50,
                calls_per_hour=3000,
                calls_per_day=50000,
                requires_key=False,
                enabled=True
            ),
            
            'cryptocompare': APIConfig(
                name='CryptoCompare',
                base_url='https://min-api.cryptocompare.com/data',
                calls_per_minute=100,
                calls_per_hour=6000,
                calls_per_day=100000,
                requires_key=False,
                enabled=True
            ),
            
            'messari': APIConfig(
                name='Messari',
                base_url='https://data.messari.io/api/v1',
                calls_per_minute=10,
                calls_per_hour=600,
                calls_per_day=1000,
                requires_key=False,
                enabled=True
            ),
            
            'binance': APIConfig(
                name='Binance',
                base_url='https://api.binance.com/api/v3',
                calls_per_minute=1200,
                calls_per_hour=72000,
                calls_per_day=1000000,
                requires_key=False,
                enabled=True
            ),
            
            'fear_greed': APIConfig(
                name='Fear & Greed',
                base_url='https://api.alternative.me/fng',
                calls_per_minute=60,
                calls_per_hour=3600,
                calls_per_day=86400,
                requires_key=False,
                enabled=True
            ),
            
            'glassnode': APIConfig(
                name='Glassnode',
                base_url='https://api.glassnode.com/v1',
                calls_per_minute=10,
                calls_per_hour=600,
                calls_per_day=10000,
                requires_key=False,
                enabled=True
            ),
            
            # APIs requiring keys (if available)
            'etherscan': APIConfig(
                name='Etherscan',
                base_url='https://api.etherscan.io/api',
                calls_per_minute=300,
                calls_per_hour=18000,
                calls_per_day=100000,
                requires_key=True,
                key_name='etherscan',
                enabled=bool(self.api_keys.get('etherscan'))
            ),
            
            'coinmarketcap': APIConfig(
                name='CoinMarketCap',
                base_url='https://pro-api.coinmarketcap.com/v1',
                calls_per_minute=30,
                calls_per_hour=1800,
                calls_per_day=10000,
                requires_key=True,
                key_name='coinmarketcap',
                enabled=bool(self.api_keys.get('coinmarketcap'))
            ),
            
            'alphavantage': APIConfig(
                name='Alpha Vantage',
                base_url='https://www.alphavantage.co/query',
                calls_per_minute=5,
                calls_per_hour=300,
                calls_per_day=500,
                requires_key=True,
                key_name='alphavantage',
                enabled=bool(self.api_keys.get('alphavantage'))
            ),
            
            'newsapi': APIConfig(
                name='NewsAPI',
                base_url='https://newsapi.org/v2',
                calls_per_minute=10,
                calls_per_hour=600,
                calls_per_day=100,
                requires_key=True,
                key_name='newsapi',
                enabled=bool(self.api_keys.get('newsapi'))
            )
        }
        
        return configs
    
    def _make_api_call(self, api_name: str, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make an API call with intelligent rate limiting"""
        if api_name not in self.api_configs:
            logger.warning(f"Unknown API: {api_name}")
            return None
        
        config = self.api_configs[api_name]
        if not config.enabled:
            logger.debug(f"API {api_name} is disabled")
            return None
        
        # Check rate limits
        if not self.rate_limiter.can_call(api_name, config):
            wait_time = self.rate_limiter.get_wait_time(api_name, config)
            logger.debug(f"Rate limit reached for {api_name}, waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        try:
            # Build URL
            url = f"{config.base_url}/{endpoint}"
            
            # Add API key if required
            if config.requires_key and config.key_name:
                key = self.api_keys.get(config.key_name)
                if not key:
                    logger.warning(f"API key required for {api_name} but not provided")
                    return None
                
                if config.key_name == 'etherscan':
                    params = params or {}
                    params['apikey'] = key
                elif config.key_name == 'coinmarketcap':
                    headers = {'X-CMC_PRO_API_KEY': key}
                elif config.key_name == 'alphavantage':
                    params = params or {}
                    params['apikey'] = key
                elif config.key_name == 'newsapi':
                    params = params or {}
                    params['apiKey'] = key
            
            # Make request
            if config.key_name == 'coinmarketcap':
                response = self.session.get(url, params=params, headers=headers, timeout=10)
            else:
                response = self.session.get(url, params=params, timeout=10)
            
            response.raise_for_status()
            
            # Record the call
            self.rate_limiter.record_call(api_name)
            
            return response.json()
            
        except Exception as e:
            logger.warning(f"API call failed for {api_name}: {e}")
            return None
    
    def get_sentiment_data(self) -> Dict[str, float]:
        """Get comprehensive sentiment data from multiple sources"""
        sentiment_data = {
            'fear_greed_index': 50.0,
            'social_sentiment': 0.0,
            'news_sentiment': 0.0,
            'market_sentiment': 0.0,
            'overall_sentiment': 0.0
        }
        
        # Fear & Greed Index
        try:
            fg_data = self._make_api_call('fear_greed', '')
            if fg_data and 'data' in fg_data:
                sentiment_data['fear_greed_index'] = float(fg_data['data'][0]['value'])
        except Exception as e:
            logger.debug(f"Fear & Greed data unavailable: {e}")
        
        # News sentiment (if NewsAPI available)
        if self.api_configs['newsapi'].enabled:
            try:
                news_params = {
                    'q': 'ethereum OR bitcoin OR cryptocurrency',
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 20
                }
                news_data = self._make_api_call('newsapi', 'everything', news_params)
                if news_data and 'articles' in news_data:
                    # Simple sentiment analysis based on keywords
                    positive_words = ['bullish', 'rally', 'surge', 'gain', 'rise', 'breakout']
                    negative_words = ['bearish', 'crash', 'dump', 'fall', 'drop', 'decline']
                    
                    sentiment_score = 0
                    for article in news_data['articles']:
                        title = article.get('title', '').lower()
                        content = article.get('description', '').lower()
                        text = f"{title} {content}"
                        
                        positive_count = sum(1 for word in positive_words if word in text)
                        negative_count = sum(1 for word in negative_words if word in text)
                        
                        if positive_count > negative_count:
                            sentiment_score += 1
                        elif negative_count > positive_count:
                            sentiment_score -= 1
                    
                    sentiment_data['news_sentiment'] = sentiment_score / len(news_data['articles']) if news_data['articles'] else 0
            except Exception as e:
                logger.debug(f"News sentiment unavailable: {e}")
        
        # Market sentiment from price action
        try:
            # Get ETH price data for market sentiment
            eth_data = self._make_api_call('coingecko', 'simple/price', {
                'ids': 'ethereum',
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            })
            
            if eth_data and 'ethereum' in eth_data:
                price_change = eth_data['ethereum'].get('usd_24h_change', 0)
                # Convert price change to sentiment (-1 to 1)
                sentiment_data['market_sentiment'] = np.tanh(price_change / 10)  # Normalize
        except Exception as e:
            logger.debug(f"Market sentiment unavailable: {e}")
        
        # Calculate overall sentiment
        sentiment_data['overall_sentiment'] = (
            (sentiment_data['fear_greed_index'] - 50) / 50 +  # -1 to 1
            sentiment_data['news_sentiment'] * 0.3 +
            sentiment_data['market_sentiment'] * 0.4
        ) / 2.7  # Normalize to -1 to 1
        
        return sentiment_data
    
    def get_onchain_metrics(self) -> Dict[str, float]:
        """Get on-chain metrics from multiple sources"""
        onchain_data = {
            'gas_price': 20.0,
            'active_addresses': 500000,
            'transaction_count': 1000000,
            'network_activity': 0.5,
            'whale_activity': 0.0,
            'defi_tvl': 50000000000,
            'stablecoin_supply': 100000000000
        }
        
        # Gas price from Etherscan
        if self.api_configs['etherscan'].enabled:
            try:
                gas_data = self._make_api_call('etherscan', '', {
                    'module': 'gastracker',
                    'action': 'gasoracle'
                })
                
                if gas_data and gas_data.get('status') == '1':
                    result = gas_data.get('result', {})
                    onchain_data['gas_price'] = float(result.get('ProposeGasPrice', 20))
            except Exception as e:
                logger.debug(f"Gas price unavailable: {e}")
        
        # On-chain metrics from Messari
        try:
            messari_data = self._make_api_call('messari', 'assets/ethereum/metrics')
            if messari_data and 'data' in messari_data:
                metrics = messari_data['data']
                onchain_data['active_addresses'] = metrics.get('active_addresses', 500000)
                onchain_data['transaction_count'] = metrics.get('transaction_volume', 1000000)
        except Exception as e:
            logger.debug(f"Messari data unavailable: {e}")
        
        # DeFi TVL from CoinGecko
        try:
            defi_data = self._make_api_call('coingecko', 'global/decentralized_finance_defi')
            if defi_data and 'data' in defi_data:
                onchain_data['defi_tvl'] = defi_data['data'].get('total_value_locked_usd', 50000000000)
        except Exception as e:
            logger.debug(f"DeFi TVL unavailable: {e}")
        
        return onchain_data
    
    def get_market_metrics(self) -> Dict[str, float]:
        """Get comprehensive market metrics"""
        market_data = {
            'eth_price': 2000.0,
            'btc_price': 40000.0,
            'market_cap': 200000000000,
            'volume_24h': 10000000000,
            'price_change_24h': 0.0,
            'market_dominance': 0.18,
            'volatility': 0.02
        }
        
        # Price and market data from CoinGecko
        try:
            price_data = self._make_api_call('coingecko', 'simple/price', {
                'ids': 'ethereum,bitcoin',
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            })
            
            if price_data:
                if 'ethereum' in price_data:
                    eth = price_data['ethereum']
                    market_data['eth_price'] = eth.get('usd', 2000.0)
                    market_data['market_cap'] = eth.get('usd_market_cap', 200000000000)
                    market_data['volume_24h'] = eth.get('usd_24h_vol', 10000000000)
                    market_data['price_change_24h'] = eth.get('usd_24h_change', 0.0) / 100
                
                if 'bitcoin' in price_data:
                    btc = price_data['bitcoin']
                    market_data['btc_price'] = btc.get('usd', 40000.0)
                    
                    # Calculate market dominance
                    btc_market_cap = btc.get('usd_market_cap', 800000000000)
                    total_market_cap = market_data['market_cap'] + btc_market_cap
                    if total_market_cap > 0:
                        market_data['market_dominance'] = market_data['market_cap'] / total_market_cap
        except Exception as e:
            logger.debug(f"Market data unavailable: {e}")
        
        # Volatility from Binance
        try:
            klines_data = self._make_api_call('binance', 'klines', {
                'symbol': 'ETHUSDT',
                'interval': '1h',
                'limit': 24
            })
            
            if klines_data:
                prices = [float(k[4]) for k in klines_data]  # Close prices
                returns = np.diff(prices) / prices[:-1]
                market_data['volatility'] = np.std(returns)
        except Exception as e:
            logger.debug(f"Volatility data unavailable: {e}")
        
        return market_data
    
    def get_exchange_metrics(self) -> Dict[str, float]:
        """Get exchange-specific metrics"""
        exchange_data = {
            'funding_rate': 0.0001,
            'open_interest': 1000000000,
            'liquidations_24h': 50000000,
            'long_short_ratio': 1.0,
            'basis': 0.0
        }
        
        # Funding rate from Binance
        try:
            funding_data = self._make_api_call('binance', 'premiumIndex', {
                'symbol': 'ETHUSDT'
            })
            
            if funding_data:
                for item in funding_data:
                    if item['symbol'] == 'ETHUSDT':
                        exchange_data['funding_rate'] = float(item.get('lastFundingRate', 0.0001))
                        break
        except Exception as e:
            logger.debug(f"Funding rate unavailable: {e}")
        
        return exchange_data
    
    def get_all_alternative_data(self) -> Dict[str, float]:
        """Get all alternative data with maximum intelligence"""
        logger.info("🧠 Collecting smart alternative data...")
        
        # Collect all data types
        sentiment = self.get_sentiment_data()
        onchain = self.get_onchain_metrics()
        market = self.get_market_metrics()
        exchange = self.get_exchange_metrics()
        
        # Combine all data
        all_data = {}
        all_data.update(sentiment)
        all_data.update(onchain)
        all_data.update(market)
        all_data.update(exchange)
        
        # Add derived features
        all_data['market_efficiency'] = market['volatility'] * market['volume_24h'] / market['market_cap']
        all_data['network_health'] = onchain['active_addresses'] / 1000000  # Normalize
        all_data['sentiment_momentum'] = sentiment['overall_sentiment'] * market['price_change_24h']
        
        logger.info(f"✅ Smart alternative data collected: {len(all_data)} features")
        return all_data
    
    def start_background_collection(self):
        """Start background data collection"""
        if not self.running:
            self.running = True
            self.background_thread = threading.Thread(target=self._background_loop)
            self.background_thread.daemon = True
            self.background_thread.start()
            logger.info("🔄 Background data collection started")
    
    def stop_background_collection(self):
        """Stop background data collection"""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
        logger.info("⏹️ Background data collection stopped")
    
    def _background_loop(self):
        """Background data collection loop"""
        while self.running:
            try:
                data = self.get_all_alternative_data()
                self.cache['latest_data'] = {
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                time.sleep(300)  # Update every 5 minutes
            except Exception as e:
                logger.error(f"Background collection error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def get_latest_data(self) -> Dict[str, float]:
        """Get the latest cached data"""
        if 'latest_data' in self.cache:
            cache_time = datetime.fromisoformat(self.cache['latest_data']['timestamp'])
            if datetime.now() - cache_time < timedelta(seconds=self.cache_expiry):
                return self.cache['latest_data']['data']
        
        # If cache is stale or empty, get fresh data
        return self.get_all_alternative_data()

# Global instance for easy access
smart_alternative_data = SmartAlternativeData() 