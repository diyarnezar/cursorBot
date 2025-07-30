#!/usr/bin/env python3
"""
CRYPTO-SPECIFIC FEATURES MODULE - ETH/FDUSD OPTIMIZED
Project Hyperion - Advanced Crypto Trading Intelligence

This module provides comprehensive crypto-specific features optimized for ETH/FDUSD trading on Binance,
including funding rates, open interest, liquidations, whale activity, and advanced crypto indicators.
"""

import pandas as pd
import numpy as np
import logging
import time
import json
import requests
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import deque
import threading
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoFeatures:
    """Advanced crypto-specific features for ETH/FDUSD trading"""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Use spot API for ETHFDUSD (not futures)
        self.binance_spot = "https://api.binance.com/api/v3"
        self.binance_futures = "https://fapi.binance.com/fapi/v1"  # Only for actual futures pairs
        
        # Feature history for calculations
        self.feature_history = {
            'funding_rates': [],
            'open_interest': [],
            'liquidations': [],
            'order_book_imbalance': [],
            'whale_activity': [],
            'taker_maker_volume': []
        }
        
        # Cache for API calls
        self.cache = {}
        self.cache_timeout = 60  # 1 minute cache
        self.cache_lock = threading.Lock()
        
        logger.info("🚀 Crypto Features initialized for ETH/FDUSD trading")
    
    def get_funding_rate(self, symbol: str = "ETHFDUSD") -> Dict[str, float]:
        """Get funding rate data for ETH/FDUSD (spot trading - no funding rate)"""
        try:
            cache_key = f"funding_rate_{symbol}"
            cached = self._get_cached(cache_key)
            if cached:
                return cached
            
            # ETHFDUSD is a spot trading pair - no funding rate
            # Return zero values for spot trading
            result = {
                'funding_rate': 0.0,
                'next_funding_time': 0,
                'funding_rate_annualized': 0.0,
                'funding_rate_impact': 0.0
            }
            
            self._cache_result(cache_key, result)
            self.feature_history['funding_rates'].append(result)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get funding rate: {e}")
            return {
                'funding_rate': 0.0,
                'next_funding_time': 0,
                'funding_rate_annualized': 0.0,
                'funding_rate_impact': 0.0
            }
    
    def get_open_interest(self, symbol: str = "ETHFDUSD") -> Dict[str, float]:
        """Get open interest data for ETH/FDUSD (spot trading - no open interest)"""
        try:
            cache_key = f"open_interest_{symbol}"
            cached = self._get_cached(cache_key)
            if cached:
                return cached
            
            # ETHFDUSD is a spot trading pair - no open interest
            # Return zero values for spot trading
            result = {
                'open_interest': 0.0,
                'open_interest_change': 0.0,
                'open_interest_ma': 0.0,
                'open_interest_volatility': 0.0
            }
            
            self._cache_result(cache_key, result)
            self.feature_history['open_interest'].append(result)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get open interest: {e}")
            return {
                'open_interest': 0.0,
                'open_interest_change': 0.0,
                'open_interest_ma': 0.0,
                'open_interest_volatility': 0.0
            }
    
    def get_liquidations(self, symbol: str = "ETHFDUSD") -> Dict[str, float]:
        """Get liquidation data for ETH/FDUSD (spot trading - estimate from large trades)"""
        try:
            cache_key = f"liquidations_{symbol}"
            cached = self._get_cached(cache_key)
            if cached:
                return cached
            
            # Get recent trades to estimate liquidations (spot trading)
            url = f"{self.binance_spot}/trades"
            params = {'symbol': symbol, 'limit': 1000}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            trades = response.json()
            
            # Analyze large trades that might be liquidations
            large_trades = [t for t in trades if float(t['qty']) > 10]  # Large trades > 10 ETH
            
            long_liquidations = sum(1 for t in large_trades if t['isBuyerMaker'])
            short_liquidations = sum(1 for t in large_trades if not t['isBuyerMaker'])
            
            total_liquidations = long_liquidations + short_liquidations
            liquidation_ratio = total_liquidations / len(trades) if trades else 0.0
            
            result = {
                'long_liquidations': long_liquidations,
                'short_liquidations': short_liquidations,
                'total_liquidations': total_liquidations,
                'liquidation_ratio': liquidation_ratio,
                'liquidation_imbalance': (long_liquidations - short_liquidations) / max(total_liquidations, 1),
                'liquidation_impact': self._calculate_liquidation_impact(total_liquidations, liquidation_ratio)
            }
            
            self._cache_result(cache_key, result)
            self.feature_history['liquidations'].append(result)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get liquidations: {e}")
            return {
                'long_liquidations': 0,
                'short_liquidations': 0,
                'total_liquidations': 0,
                'liquidation_ratio': 0.0,
                'liquidation_imbalance': 0.0,
                'liquidation_impact': 0.0
            }
    
    def get_order_book_imbalance(self, symbol: str = "ETHFDUSD", depth: int = 20) -> Dict[str, float]:
        """Get order book imbalance and depth analysis"""
        try:
            cache_key = f"order_book_{symbol}_{depth}"
            cached = self._get_cached(cache_key)
            if cached:
                return cached
            
            url = f"{self.binance_spot}/depth"
            params = {'symbol': symbol, 'limit': depth}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            # Calculate bid/ask volumes
            bid_volume = sum(float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[1]) for ask in asks)
            total_volume = bid_volume + ask_volume
            
            # Order book imbalance
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0.0
            
            # Depth analysis
            bid_depth = self._calculate_depth_metrics(bids, 'bid')
            ask_depth = self._calculate_depth_metrics(asks, 'ask')
            
            # Spread analysis
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0.0
            
            result = {
                'order_book_imbalance': imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': total_volume,
                'bid_depth_1': bid_depth.get('depth_1', 0),
                'bid_depth_5': bid_depth.get('depth_5', 0),
                'bid_depth_10': bid_depth.get('depth_10', 0),
                'ask_depth_1': ask_depth.get('depth_1', 0),
                'ask_depth_5': ask_depth.get('depth_5', 0),
                'ask_depth_10': ask_depth.get('depth_10', 0),
                'spread': spread,
                'spread_bps': spread * 10000,  # Basis points
                'depth_imbalance': (bid_depth.get('depth_10', 0) - ask_depth.get('depth_10', 0)) / max(bid_depth.get('depth_10', 0) + ask_depth.get('depth_10', 0), 1)
            }
            
            self._cache_result(cache_key, result)
            self.feature_history['order_book_imbalance'].append(result)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get order book: {e}")
            return {
                'order_book_imbalance': 0.0,
                'bid_volume': 0.0,
                'ask_volume': 0.0,
                'total_volume': 0.0,
                'bid_depth_1': 0.0,
                'bid_depth_5': 0.0,
                'bid_depth_10': 0.0,
                'ask_depth_1': 0.0,
                'ask_depth_5': 0.0,
                'ask_depth_10': 0.0,
                'spread': 0.0,
                'spread_bps': 0.0,
                'depth_imbalance': 0.0
            }
    
    def get_whale_activity(self, symbol: str = "ETHFDUSD") -> Dict[str, float]:
        """Detect whale activity and large trades"""
        try:
            cache_key = f"whale_activity_{symbol}"
            cached = self._get_cached(cache_key)
            if cached:
                return cached
            
            # Get recent trades
            url = f"{self.binance_spot}/trades"
            params = {'symbol': symbol, 'limit': 1000}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            trades = response.json()
            
            # Analyze whale activity
            whale_threshold = 50  # ETH
            large_threshold = 20   # ETH
            
            whale_trades = [t for t in trades if float(t['qty']) >= whale_threshold]
            large_trades = [t for t in trades if float(t['qty']) >= large_threshold]
            
            whale_volume = sum(float(t['qty']) for t in whale_trades)
            large_volume = sum(float(t['qty']) for t in large_trades)
            total_volume = sum(float(t['qty']) for t in trades)
            
            whale_ratio = whale_volume / total_volume if total_volume > 0 else 0.0
            large_ratio = large_volume / total_volume if total_volume > 0 else 0.0
            
            # Calculate whale activity score
            whale_score = self._calculate_whale_activity_score(whale_trades, large_trades, len(trades))
            
            result = {
                'whale_trades': len(whale_trades),
                'large_trades': len(large_trades),
                'whale_volume': whale_volume,
                'large_volume': large_volume,
                'whale_ratio': whale_ratio,
                'large_ratio': large_ratio,
                'whale_activity_score': whale_score,
                'whale_imbalance': (sum(1 for t in whale_trades if t['isBuyerMaker']) - 
                                  sum(1 for t in whale_trades if not t['isBuyerMaker'])) / max(len(whale_trades), 1)
            }
            
            self._cache_result(cache_key, result)
            self.feature_history['whale_activity'].append(result)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get whale activity: {e}")
            return {
                'whale_trades': 0,
                'large_trades': 0,
                'whale_volume': 0.0,
                'large_volume': 0.0,
                'whale_ratio': 0.0,
                'large_ratio': 0.0,
                'whale_activity_score': 0.0,
                'whale_imbalance': 0.0
            }
    
    def get_taker_maker_volume(self, symbol: str = "ETHFDUSD") -> Dict[str, float]:
        """Get taker/maker volume analysis"""
        try:
            cache_key = f"taker_maker_{symbol}"
            cached = self._get_cached(cache_key)
            if cached:
                return cached
            
            # Get 24hr ticker for taker/maker volume
            url = f"{self.binance_spot}/ticker/24hr"
            params = {'symbol': symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract volume data
            quote_volume = float(data.get('quoteVolume', 0))
            count = int(data.get('count', 0))
            
            # Estimate taker/maker from recent trades
            trades_url = f"{self.binance_spot}/trades"
            trades_params = {'symbol': symbol, 'limit': 1000}
            
            trades_response = self.session.get(trades_url, params=trades_params, timeout=10)
            trades_response.raise_for_status()
            trades = trades_response.json()
            
            taker_volume = sum(float(t['qty']) for t in trades if not t['isBuyerMaker'])
            maker_volume = sum(float(t['qty']) for t in trades if t['isBuyerMaker'])
            total_trade_volume = taker_volume + maker_volume
            
            taker_ratio = taker_volume / total_trade_volume if total_trade_volume > 0 else 0.5
            maker_ratio = maker_volume / total_trade_volume if total_trade_volume > 0 else 0.5
            
            result = {
                'taker_volume': taker_volume,
                'maker_volume': maker_volume,
                'taker_ratio': taker_ratio,
                'maker_ratio': maker_ratio,
                'taker_maker_imbalance': taker_ratio - maker_ratio,
                'volume_imbalance': (taker_volume - maker_volume) / max(total_trade_volume, 1),
                'trade_count': count,
                'quote_volume': quote_volume
            }
            
            self._cache_result(cache_key, result)
            self.feature_history['taker_maker_volume'].append(result)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get taker/maker volume: {e}")
            return {
                'taker_volume': 0.0,
                'maker_volume': 0.0,
                'taker_ratio': 0.5,
                'maker_ratio': 0.5,
                'taker_maker_imbalance': 0.0,
                'volume_imbalance': 0.0,
                'trade_count': 0,
                'quote_volume': 0.0
            }
    
    def get_all_crypto_features(self, symbol: str = "ETHFDUSD") -> Dict[str, float]:
        """Get all crypto-specific features in one call"""
        try:
            # Collect all features
            funding_data = self.get_funding_rate(symbol)
            oi_data = self.get_open_interest(symbol)
            liquidation_data = self.get_liquidations(symbol)
            order_book_data = self.get_order_book_imbalance(symbol)
            whale_data = self.get_whale_activity(symbol)
            taker_maker_data = self.get_taker_maker_volume(symbol)
            
            # Combine all features
            all_features = {}
            all_features.update(funding_data)
            all_features.update(oi_data)
            all_features.update(liquidation_data)
            all_features.update(order_book_data)
            all_features.update(whale_data)
            all_features.update(taker_maker_data)
            
            # Add composite features
            all_features.update(self._calculate_composite_features(
                funding_data, oi_data, liquidation_data, 
                order_book_data, whale_data, taker_maker_data
            ))
            
            return all_features
            
        except Exception as e:
            logger.error(f"Failed to get all crypto features: {e}")
            return {}
    
    def _calculate_funding_impact(self, funding_rate: float) -> float:
        """Calculate funding rate impact on price"""
        # Higher funding rates create pressure for mean reversion
        return -funding_rate * 100  # Negative because high funding = short pressure
    
    def _calculate_liquidation_impact(self, total_liquidations: int, liquidation_ratio: float) -> float:
        """Calculate liquidation impact on price"""
        # More liquidations = higher volatility and potential price impact
        return min(total_liquidations * 0.1, 10.0)  # Cap at 10
    
    def _calculate_whale_activity_score(self, whale_trades: List, large_trades: List, total_trades: int) -> float:
        """Calculate whale activity score"""
        if total_trades == 0:
            return 0.0
        
        whale_ratio = len(whale_trades) / total_trades
        large_ratio = len(large_trades) / total_trades
        
        return (whale_ratio * 10 + large_ratio * 5) * 100  # Scale to 0-100
    
    def _calculate_depth_metrics(self, orders: List, side: str) -> Dict[str, float]:
        """Calculate order book depth metrics"""
        if not orders:
            return {'depth_1': 0.0, 'depth_5': 0.0, 'depth_10': 0.0}
        
        depth_1 = sum(float(order[1]) for order in orders[:1])
        depth_5 = sum(float(order[1]) for order in orders[:5])
        depth_10 = sum(float(order[1]) for order in orders[:10])
        
        return {
            'depth_1': depth_1,
            'depth_5': depth_5,
            'depth_10': depth_10
        }
    
    def _calculate_ma_from_history(self, feature_type: str, current_value: float) -> float:
        """Calculate moving average from feature history"""
        history = list(self.feature_history[feature_type])
        if not history:
            return current_value
        
        values = [h.get('open_interest', current_value) for h in history[-20:]]  # Last 20 values
        return sum(values) / len(values)
    
    def _calculate_volatility_from_history(self, feature_type: str, current_value: float) -> float:
        """Calculate volatility from feature history"""
        history = list(self.feature_history[feature_type])
        if len(history) < 2:
            return 0.0
        
        values = [h.get('open_interest', current_value) for h in history[-20:]]
        if len(values) < 2:
            return 0.0
        
        return float(np.std(values))
    
    def _calculate_composite_features(self, funding_data: Dict, oi_data: Dict, 
                                    liquidation_data: Dict, order_book_data: Dict,
                                    whale_data: Dict, taker_maker_data: Dict) -> Dict[str, float]:
        """Calculate composite crypto features"""
        composite = {}
        
        # Funding rate + Open Interest interaction
        composite['funding_oi_pressure'] = (
            funding_data.get('funding_rate', 0) * oi_data.get('open_interest_change', 0)
        )
        
        # Liquidation + Whale activity
        composite['liquidation_whale_impact'] = (
            liquidation_data.get('total_liquidations', 0) * whale_data.get('whale_activity_score', 0) / 100
        )
        
        # Order book + Taker/Maker imbalance
        composite['order_flow_imbalance'] = (
            order_book_data.get('order_book_imbalance', 0) * taker_maker_data.get('taker_imbalance', 0)
        )
        
        # Volatility prediction
        composite['volatility_predictor'] = (
            abs(funding_data.get('funding_rate', 0)) * 100 +
            liquidation_data.get('liquidation_ratio', 0) * 50 +
            whale_data.get('whale_activity_score', 0) / 10
        )
        
        # Trend strength
        composite['trend_strength'] = (
            abs(order_book_data.get('order_book_imbalance', 0)) * 100 +
            abs(taker_maker_data.get('taker_imbalance', 0)) * 50 +
            abs(whale_data.get('whale_imbalance', 0)) * 25
        )
        
        return composite
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached result if not expired"""
        with self.cache_lock:
            if key in self.cache:
                timestamp, data = self.cache[key]
                if time.time() - timestamp < self.cache_timeout:
                    return data
                else:
                    del self.cache[key]
            return None
    
    def _cache_result(self, key: str, data: Dict):
        """Cache result with timestamp"""
        with self.cache_lock:
            self.cache[key] = (time.time(), data)
    
    def clear_cache(self):
        """Clear all cached data"""
        with self.cache_lock:
            self.cache.clear()
    
    def get_feature_names(self) -> List[str]:
        """Get list of all available crypto feature names"""
        return [
            # Funding rate features
            'funding_rate', 'next_funding_time', 'funding_rate_annualized', 'funding_rate_impact',
            
            # Open interest features
            'open_interest', 'open_interest_change', 'open_interest_ma', 'open_interest_volatility',
            
            # Liquidation features
            'long_liquidations', 'short_liquidations', 'total_liquidations', 'liquidation_ratio',
            'liquidation_imbalance', 'liquidation_impact',
            
            # Order book features
            'order_book_imbalance', 'bid_volume', 'ask_volume', 'total_volume',
            'bid_depth_1', 'bid_depth_5', 'bid_depth_10',
            'ask_depth_1', 'ask_depth_5', 'ask_depth_10',
            'spread', 'spread_bps', 'depth_imbalance',
            
            # Whale activity features
            'whale_trades', 'large_trades', 'whale_volume', 'large_volume',
            'whale_ratio', 'large_ratio', 'whale_imbalance', 'whale_activity_score',
            
            # Taker/Maker features
            'taker_volume', 'maker_volume', 'taker_ratio', 'maker_ratio',
            'taker_maker_imbalance', 'volume_imbalance', 'trade_count', 'quote_volume',
            
            # Composite features
            'funding_oi_pressure', 'liquidation_whale_impact', 'order_flow_imbalance',
            'volatility_predictor', 'trend_strength'
        ] 