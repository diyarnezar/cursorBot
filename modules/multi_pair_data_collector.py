#!/usr/bin/env python3
"""
Multi-Pair Data Collector
Advanced data collection system for all 26 FDUSD pairs with intelligent caching and optimization
"""

import os
import sys
import json
import logging
import time
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_ingestion import fetch_klines, fetch_ticker_24hr, fetch_order_book
from modules.api_connection_manager import APIConnectionManager
from modules.smart_data_collector import SmartDataCollector

warnings.filterwarnings('ignore')

class MultiPairDataCollector:
    """
    Advanced multi-pair data collector with intelligent optimization
    Collects data for all 26 FDUSD pairs efficiently
    """
    
    def __init__(self, config_path: str = 'config.json'):
        self.logger = logging.getLogger(__name__)
        self.config = self.load_config(config_path)
        
        # Multi-pair configuration
        self.multi_pair_config = self.config.get('trading_parameters', {}).get('multi_pair_trading', {})
        self.all_pairs = self.multi_pair_config.get('all_pairs', [])
        self.asset_clusters = self.multi_pair_config.get('asset_clusters', {})
        self.cluster_weights = self.multi_pair_config.get('cluster_weights', {})
        
        # Initialize API manager
        self.api_manager = APIConnectionManager()
        
        # Data storage
        self.data_cache = {}
        self.last_update = {}
        self.cache_expiry = 300  # 5 minutes
        
        # Performance tracking
        self.collection_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'last_collection_time': None
        }
        
        # Threading
        self.lock = threading.Lock()
        self.running = False
        
        self.logger.info(f"🚀 Multi-Pair Data Collector initialized for {len(self.all_pairs)} pairs")
        self.logger.info(f"   Pairs: {self.all_pairs}")
        self.logger.info(f"   Clusters: {list(self.asset_clusters.keys())}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def get_pair_cluster(self, pair: str) -> str:
        """Get the cluster for a given pair"""
        asset = pair.replace('FDUSD', '')
        for cluster, assets in self.asset_clusters.items():
            if asset in assets:
                return cluster
        return 'unknown'
    
    def collect_all_pairs_data(self, interval: str = '1m', limit: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Collect data for all pairs efficiently with rate limiting
        """
        self.logger.info(f"📊 Collecting data for all {len(self.all_pairs)} pairs with rate limiting...")
        
        # Use the optimized collector instead of direct API calls
        from modules.optimized_data_collector import optimized_collector
        
        start_time = time.time()
        
        # Use optimized collection with rate limiting
        results = optimized_collector.collect_all_pairs_data(self.all_pairs, days=limit/1440)  # Convert minutes to days
        
        # Update statistics
        collection_time = time.time() - start_time
        self.collection_stats['last_collection_time'] = datetime.now()
        self.collection_stats['total_requests'] += len(self.all_pairs)
        self.collection_stats['successful_requests'] += len(results)
        self.collection_stats['failed_requests'] += len(self.all_pairs) - len(results)
        
        # Get rate limiting statistics
        rate_stats = optimized_collector.get_collection_stats()
        self.logger.info(f"📈 Collection complete: {len(results)}/{len(self.all_pairs)} pairs in {collection_time:.2f}s")
        self.logger.info(f"🛡️ Rate limiting stats: {rate_stats['rate_limiter_stats']}")
        
        return results
    
    def _collect_pair_data(self, pair: str, interval: str = '1m', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Collect data for a single pair with caching and rate limiting
        """
        try:
            # Check cache first
            cache_key = f"{pair}_{interval}_{limit}"
            if cache_key in self.data_cache:
                last_update = self.data_cache[cache_key]
                if time.time() - last_update < self.cache_expiry:
                    self.collection_stats['cache_hits'] += 1
                    return self.data_cache[cache_key]
            
            # Use optimized collector with rate limiting
            from modules.optimized_data_collector import optimized_collector
            
            # Convert limit to days for optimized collector
            days = limit / 1440  # Convert minutes to days
            
            # Apply rate limiting
            optimized_collector.rate_limiter.wait_if_needed()
            
            # Calculate time range based on limit (reduced from 1000 to 100)
            end_time = datetime.now()
            if interval == '1m':
                start_time = end_time - timedelta(minutes=limit)
            elif interval == '5m':
                start_time = end_time - timedelta(minutes=limit * 5)
            elif interval == '15m':
                start_time = end_time - timedelta(minutes=limit * 15)
            elif interval == '1h':
                start_time = end_time - timedelta(hours=limit)
            elif interval == '4h':
                start_time = end_time - timedelta(hours=limit * 4)
            elif interval == '1d':
                start_time = end_time - timedelta(days=limit)
            else:
                start_time = end_time - timedelta(minutes=limit)
            
            # Use optimized data collection
            data = optimized_collector.collect_pair_data(pair, days)
            
            if data is not None:
                # Add pair and cluster information
                data['pair'] = pair
                data['asset'] = pair.replace('FDUSD', '')
                data['cluster'] = self.get_pair_cluster(pair)
                
                # Cache the data
                with self.lock:
                    self.data_cache[cache_key] = data
                    self.last_update[cache_key] = time.time()
                
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error collecting data for {pair}: {e}")
            return None
    
    def collect_enhanced_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Add enhanced features for all pairs
        """
        self.logger.info("🧠 Adding enhanced features for all pairs...")
        
        enhanced_data = {}
        
        for pair, df in data_dict.items():
            try:
                enhanced_df = self._add_pair_features(df, pair)
                enhanced_data[pair] = enhanced_df
                self.logger.info(f"   ✅ {pair}: {enhanced_df.shape[1]} features")
                except Exception as e:
                self.logger.error(f"   ❌ {pair}: Error adding features - {e}")
                enhanced_data[pair] = df  # Use original data if feature addition fails
        
        return enhanced_data
    
    def _add_pair_features(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Add comprehensive features for a single pair
        """
        if df.empty:
            return df
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Momentum features
        df = self._add_momentum_features(df)
        
        # Microstructure features
        df = self._add_microstructure_features(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_21'] = self._calculate_rsi(df['close'], 21)
        
        # MACD
        macd, signal, hist = self._calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        df['returns_1m'] = df['close'].pct_change()
        df['returns_5m'] = df['close'].pct_change(5)
        df['returns_15m'] = df['close'].pct_change(15)
        
        # Price levels
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open']
        
        # High-Low features
        df['hl_ratio'] = df['high'] / df['low']
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volume-price relationship
        df['volume_price_trend'] = (df['volume'] * df['returns_1m']).rolling(20).sum()
        
        # Buy/Sell pressure
        df['buy_volume'] = df['taker_buy_quote_asset_volume']
        df['sell_volume'] = df['quote_asset_volume'] - df['buy_volume']
        df['buy_sell_ratio'] = df['buy_volume'] / (df['sell_volume'] + 1e-8)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        # Rolling volatility
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['returns_1m'].rolling(period).std()
        
        # ATR (Average True Range)
        df['atr_14'] = self._calculate_atr(df, 14)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # Momentum indicators
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Trade intensity
        df['trade_intensity'] = df['number_of_trades'] / (df['volume'] + 1e-8)
        
        # Average trade size
        df['avg_trade_size'] = df['volume'] / (df['number_of_trades'] + 1e-8)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std_dev = prices.rolling(period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'total_pairs': len(self.all_pairs),
            'cache_size': len(self.data_cache),
            'collection_stats': self.collection_stats,
            'clusters': list(self.asset_clusters.keys()),
            'cluster_weights': self.cluster_weights
        }
    
    def clear_cache(self):
        """Clear data cache"""
        with self.lock:
            self.data_cache.clear()
            self.last_update.clear()
        self.logger.info("🗑️ Data cache cleared")
    
    def start_background_collection(self, interval_seconds: int = 300):
        """Start background data collection"""
        if self.running:
            self.logger.warning("Background collection already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._background_collection_loop, 
                                                args=(interval_seconds,), daemon=True)
        self.collection_thread.start()
        self.logger.info(f"🔄 Background collection started (interval: {interval_seconds}s)")
    
    def stop_background_collection(self):
        """Stop background data collection"""
        self.running = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=10)
        self.logger.info("⏹️ Background collection stopped")
    
    def _background_collection_loop(self, interval_seconds: int):
        """Background collection loop"""
        while self.running:
            try:
                self.collect_all_pairs_data()
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in background collection: {e}")
                time.sleep(60)  # Wait 1 minute on error

    def collect_advanced_data_for_pair(self, pair: str, days: float = 15.0) -> Optional[pd.DataFrame]:
        """
        Collect advanced data for a single pair with all features
        """
        try:
            # Convert pair to FDUSD format
            pair_symbol = f"{pair}FDUSD"
            
            # Calculate limit based on days (assuming 1-minute data)
            limit = int(days * 24 * 60)  # minutes per day
            limit = min(limit, 1000)  # Cap at 1000 for API limits
            
            self.logger.info(f"📊 Collecting {days} days of data for {pair_symbol}...")
            
            # Collect data for this pair
            data_dict = self.collect_all_pairs_data(interval='1m', limit=limit)
            
            if pair_symbol not in data_dict:
                self.logger.error(f"No data collected for {pair_symbol}")
                return None
            
            data = data_dict[pair_symbol]
            
            if data.empty:
                self.logger.error(f"Empty data for {pair_symbol}")
                return None
            
            # Add enhanced features
            enhanced_data = self.collect_enhanced_features({pair_symbol: data})
            
            if pair_symbol not in enhanced_data:
                self.logger.error(f"Feature enhancement failed for {pair_symbol}")
                return None
            
            final_data = enhanced_data[pair_symbol]
            
            self.logger.info(f"✅ {pair_symbol}: {len(final_data)} data points, {len(final_data.columns)} features")
            
            return final_data
            
        except Exception as e:
            self.logger.error(f"Error collecting advanced data for {pair}: {e}")
            return None
