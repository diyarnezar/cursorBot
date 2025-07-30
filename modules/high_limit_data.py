#!/usr/bin/env python3
# High-Limit Alternative Data Module
# Uses free APIs with generous rate limits for smart trading decisions

import requests
import logging
import time
import json
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class HighLimitDataCollector:
    """
    Collects alternative data from high-limit free APIs:
    - CoinGecko (50 calls/minute)
    - CryptoCompare (100k calls/month)
    - Messari (1000 calls/day)
    - Glassnode (10 calls/minute)
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Project-Hyperion-Trading-Bot/1.0'
        })
        
        # Rate limiting
        self.last_call = {
            'coingecko': 0,
            'cryptocompare': 0,
            'messari': 0,
            'glassnode': 0
        }
        
        # Minimum intervals between calls (in seconds)
        self.min_intervals = {
            'coingecko': 1.2,  # 50 calls/minute = 1.2 seconds between calls
            'cryptocompare': 0.1,  # Very high limit
            'messari': 86.4,  # 1000 calls/day = 86.4 seconds between calls
            'glassnode': 6.0  # 10 calls/minute = 6 seconds between calls
        }
        
        logging.info("High-Limit Data Collector initialized")
    
    def _rate_limit(self, api_name: str):
        """Implement rate limiting for each API."""
        if api_name in self.last_call:
            elapsed = time.time() - self.last_call[api_name]
            min_interval = self.min_intervals.get(api_name, 1.0)
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
            
            self.last_call[api_name] = time.time()
    
    def get_coingecko_data(self, coin_id: str = 'ethereum') -> Dict[str, Any]:
        """Get data from CoinGecko API (50 calls/minute)."""
        try:
            self._rate_limit('coingecko')
            
            # Get basic coin data
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'market_cap': data['market_data']['market_cap']['usd'],
                    'market_cap_rank': data['market_cap_rank'],
                    'total_volume': data['market_data']['total_volume']['usd'],
                    'price_change_24h': data['market_data']['price_change_percentage_24h'],
                    'price_change_7d': data['market_data']['price_change_percentage_7d'],
                    'circulating_supply': data['market_data']['circulating_supply'],
                    'total_supply': data['market_data']['total_supply'],
                    'max_supply': data['market_data']['max_supply'],
                    'community_score': data['community_score'],
                    'developer_score': data['developer_score'],
                    'liquidity_score': data['liquidity_score'],
                    'public_interest_score': data['public_interest_score'],
                    'sentiment_votes_up_percentage': data['sentiment_votes_up_percentage'],
                    'sentiment_votes_down_percentage': data['sentiment_votes_down_percentage']
                }
            else:
                logging.warning(f"CoinGecko API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logging.warning(f"CoinGecko data error: {e}")
            return {}
    
    def get_cryptocompare_data(self, symbol: str = 'ETH') -> Dict[str, Any]:
        """Get data from CryptoCompare API (100k calls/month)."""
        try:
            self._rate_limit('cryptocompare')
            
            # Get price and volume data
            url = f"https://min-api.cryptocompare.com/data/pricemultifull"
            params = {
                'fsyms': symbol,
                'tsyms': 'USD'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                raw_data = data['RAW'][symbol]['USD']
                
                return {
                    'price': raw_data['PRICE'],
                    'volume_24h': raw_data['VOLUME24HOUR'],
                    'volume_24h_to': raw_data['VOLUME24HOURTO'],
                    'market_cap': raw_data['MKTCAP'],
                    'supply': raw_data['SUPPLY'],
                    'change_24h': raw_data['CHANGE24HOUR'],
                    'change_pct_24h': raw_data['CHANGEPCT24HOUR'],
                    'high_24h': raw_data['HIGH24HOUR'],
                    'low_24h': raw_data['LOW24HOUR'],
                    'open_24h': raw_data['OPEN24HOUR']
                }
            else:
                logging.warning(f"CryptoCompare API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logging.warning(f"CryptoCompare data error: {e}")
            return {}
    
    def get_messari_data(self, asset_key: str = 'ethereum') -> Dict[str, Any]:
        """Get data from Messari API (1000 calls/day)."""
        try:
            self._rate_limit('messari')
            
            # Get asset metrics
            url = f"https://data.messari.io/api/v1/assets/{asset_key}/metrics"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                metrics = data['data']
                
                return {
                    'market_cap': metrics['market_data']['market_cap'],
                    'volume_24h': metrics['market_data']['volume_24h'],
                    'price': metrics['market_data']['price'],
                    'price_change_24h': metrics['market_data']['percent_change_24h'],
                    'roi_30d': metrics['roi_data']['percent_change_30d'],
                    'roi_60d': metrics['roi_data']['percent_change_60d'],
                    'roi_1y': metrics['roi_data']['percent_change_1y'],
                    'developer_activity': metrics['developer_data']['commit_count_4_weeks'],
                    'github_activity': metrics['developer_data']['github_activity'],
                    'reddit_subscribers': metrics['reddit_data']['subscribers'],
                    'twitter_followers': metrics['twitter_data']['followers']
                }
            else:
                # Suppress 401 errors (API key required) as they're expected for free tier
                if response.status_code != 401:
                    logging.warning(f"Messari API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logging.warning(f"Messari data error: {e}")
            return {}
    
    def get_glassnode_data(self, symbol: str = 'ETH') -> Dict[str, Any]:
        """Get data from Glassnode API (10 calls/minute)."""
        try:
            self._rate_limit('glassnode')
            
            # Get on-chain metrics
            base_url = "https://api.glassnode.com/v1/metrics"
            
            # Get active addresses
            url = f"{base_url}/addresses/active_count"
            params = {
                'a': symbol,
                'i': '24h',
                'api_key': self.api_keys.get('glassnode_api_key', '')
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                active_addresses = data[-1]['v'] if data else 0
                
                # Get transaction count
                url_tx = f"{base_url}/transactions/count"
                response_tx = self.session.get(url_tx, params=params, timeout=10)
                
                tx_count = 0
                if response_tx.status_code == 200:
                    tx_data = response_tx.json()
                    tx_count = tx_data[-1]['v'] if tx_data else 0
                
                return {
                    'active_addresses_24h': active_addresses,
                    'transaction_count_24h': tx_count,
                    'network_activity': active_addresses / 1000000 if active_addresses > 0 else 0
                }
            else:
                logging.warning(f"Glassnode API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logging.warning(f"Glassnode data error: {e}")
            return {}
    
    def get_all_data(self) -> Dict[str, float]:
        """Get all alternative data from high-limit APIs."""
        try:
            all_data = {}
            
            # Get CoinGecko data
            coingecko_data = self.get_coingecko_data()
            for key, value in coingecko_data.items():
                if isinstance(value, (int, float)):
                    all_data[f'coingecko_{key}'] = float(value)
            
            # Get CryptoCompare data
            cryptocompare_data = self.get_cryptocompare_data()
            for key, value in cryptocompare_data.items():
                if isinstance(value, (int, float)):
                    all_data[f'cryptocompare_{key}'] = float(value)
            
            # Get Messari data
            messari_data = self.get_messari_data()
            for key, value in messari_data.items():
                if isinstance(value, (int, float)):
                    all_data[f'messari_{key}'] = float(value)
            
            # Get Glassnode data
            glassnode_data = self.get_glassnode_data()
            for key, value in glassnode_data.items():
                if isinstance(value, (int, float)):
                    all_data[f'glassnode_{key}'] = float(value)
            
            # Add default values for missing data
            default_features = [
                'sentiment_score', 'news_impact', 'social_volume', 
                'funding_rate', 'liquidations', 'open_interest_change',
                'whale_activity', 'network_value', 'gas_price',
                'defi_tvl', 'stablecoin_supply', 'fear_greed_index'
            ]
            
            for feature in default_features:
                if feature not in all_data:
                    all_data[feature] = 0.0
            
            logging.info(f"Collected {len(all_data)} alternative data points from high-limit APIs")
            return all_data
            
        except Exception as e:
            logging.error(f"Error collecting high-limit data: {e}")
            return {
                'sentiment_score': 0.0,
                'news_impact': 0.0,
                'social_volume': 0.0,
                'funding_rate': 0.0,
                'liquidations': 0.0,
                'open_interest_change': 0.0,
                'whale_activity': 0.0,
                'network_value': 0.0,
                'gas_price': 20.0,
                'defi_tvl': 0.0,
                'stablecoin_supply': 0.0,
                'fear_greed_index': 50.0
            }
    
    def get_smart_indicators(self) -> Dict[str, float]:
        """Calculate smart trading indicators from collected data."""
        try:
            data = self.get_all_data()
            
            indicators = {}
            
            # Market sentiment indicator
            if 'coingecko_sentiment_votes_up_percentage' in data:
                indicators['market_sentiment'] = data['coingecko_sentiment_votes_up_percentage'] / 100
            
            # Developer activity indicator
            if 'messari_developer_activity' in data:
                indicators['developer_activity'] = min(data['messari_developer_activity'] / 1000, 1.0)
            
            # Social activity indicator
            if 'messari_reddit_subscribers' in data and 'messari_twitter_followers' in data:
                social_score = (data['messari_reddit_subscribers'] + data['messari_twitter_followers']) / 1000000
                indicators['social_activity'] = min(social_score, 1.0)
            
            # Network activity indicator
            if 'glassnode_network_activity' in data:
                indicators['network_activity'] = data['glassnode_network_activity']
            
            # Price momentum indicator
            if 'cryptocompare_change_pct_24h' in data:
                indicators['price_momentum'] = data['cryptocompare_change_pct_24h'] / 100
            
            # Volume activity indicator
            if 'cryptocompare_volume_24h' in data and 'cryptocompare_market_cap' in data:
                volume_ratio = data['cryptocompare_volume_24h'] / data['cryptocompare_market_cap']
                indicators['volume_activity'] = min(volume_ratio, 1.0)
            
            # Market dominance indicator
            if 'coingecko_market_cap_rank' in data:
                indicators['market_dominance'] = 1.0 / data['coingecko_market_cap_rank']
            
            return indicators
            
        except Exception as e:
            logging.error(f"Error calculating smart indicators: {e}")
            return {} 