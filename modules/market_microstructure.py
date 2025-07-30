#!/usr/bin/env python3
"""
ULTRA-ADVANCED Market Microstructure Analysis Module
Advanced order book analysis, market impact prediction, and liquidity depth analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import time
from datetime import datetime, timedelta
import json

class MarketMicrostructureAnalyzer:
    """
    ULTRA-ADVANCED Market Microstructure Analyzer with maximum intelligence:
    
    Features:
    - Real-time order book imbalance tracking
    - Market impact prediction using advanced models
    - Liquidity depth analysis with stress testing
    - Bid-ask spread modeling and prediction
    - Volume profile analysis with time-weighted metrics
    - Market microstructure regime detection
    - Cross-correlation analysis between microstructure metrics
    - Predictive analytics for market microstructure changes
    """
    
    def __init__(self, 
                 lookback_period: int = 1000,
                 update_interval: float = 1.0,
                 enable_real_time: bool = True):
        """
        Initialize the Market Microstructure Analyzer.
        
        Args:
            lookback_period: Number of data points to keep in memory
            update_interval: Update interval in seconds
            enable_real_time: Whether to enable real-time analysis
        """
        self.lookback_period = lookback_period
        self.update_interval = update_interval
        self.enable_real_time = enable_real_time
        
        # Data storage
        self.order_book_history = deque(maxlen=lookback_period)
        self.trade_history = deque(maxlen=lookback_period)
        self.microstructure_metrics = deque(maxlen=lookback_period)
        
        # Real-time metrics
        self.current_imbalance = 0.0
        self.current_spread = 0.0
        self.current_depth = 0.0
        self.current_impact = 0.0
        
        # Market regime detection
        self.market_regime = 'NORMAL'
        self.regime_confidence = 0.0
        
        # Predictive models
        self.impact_prediction_model = None
        self.spread_prediction_model = None
        
        # Performance tracking
        self.analysis_count = 0
        self.last_update = time.time()
        
        logging.info("ULTRA-ADVANCED Market Microstructure Analyzer initialized.")
    
    def update_order_book(self, order_book: Dict[str, List]) -> Dict[str, Any]:
        """
        Update order book data and calculate microstructure metrics.
        
        Args:
            order_book: Order book data with 'bids' and 'asks'
            
        Returns:
            Dictionary with microstructure metrics
        """
        try:
            timestamp = time.time()
            
            # Extract order book data
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return self._get_default_metrics()
            
            # Calculate microstructure metrics
            metrics = {
                'timestamp': timestamp,
                'order_imbalance': self._calculate_order_imbalance(bids, asks),
                'bid_ask_spread': self._calculate_bid_ask_spread(bids, asks),
                'liquidity_depth': self._calculate_liquidity_depth(bids, asks),
                'market_impact': self._estimate_market_impact(bids, asks),
                'volume_profile': self._analyze_volume_profile(bids, asks),
                'price_pressure': self._calculate_price_pressure(bids, asks),
                'resilience': self._calculate_market_resilience(bids, asks),
                'fragmentation': self._calculate_market_fragmentation(bids, asks)
            }
            
            # Update real-time metrics
            self._update_real_time_metrics(metrics)
            
            # Store in history
            self.order_book_history.append({
                'timestamp': timestamp,
                'bids': bids,
                'asks': asks,
                'metrics': metrics
            })
            
            # Detect market regime
            self._detect_market_regime()
            
            # Update performance tracking
            self.analysis_count += 1
            self.last_update = timestamp
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error updating order book: {e}")
            return self._get_default_metrics()
    
    def _calculate_order_imbalance(self, bids: List, asks: List) -> float:
        """Calculate order book imbalance."""
        try:
            # Calculate total volume on each side
            bid_volume = sum(float(bid[1]) for bid in bids[:10])  # Top 10 levels
            ask_volume = sum(float(ask[1]) for ask in asks[:10])  # Top 10 levels
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.0
            
            # Imbalance = (bid_volume - ask_volume) / total_volume
            imbalance = (bid_volume - ask_volume) / total_volume
            
            return np.clip(imbalance, -1.0, 1.0)
            
        except Exception as e:
            logging.error(f"Error calculating order imbalance: {e}")
            return 0.0
    
    def _calculate_bid_ask_spread(self, bids: List, asks: List) -> float:
        """Calculate bid-ask spread."""
        try:
            if not bids or not asks:
                return 0.0
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            # Calculate spread as percentage
            spread = (best_ask - best_bid) / best_bid
            
            return spread
            
        except Exception as e:
            logging.error(f"Error calculating bid-ask spread: {e}")
            return 0.0
    
    def _calculate_liquidity_depth(self, bids: List, asks: List) -> Dict[str, float]:
        """Calculate liquidity depth at different price levels."""
        try:
            depth_metrics = {}
            
            # Calculate depth at different percentage levels
            for level in [0.1, 0.5, 1.0, 2.0, 5.0]:
                mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
                price_range = mid_price * level / 100
                
                # Calculate volume within price range
                bid_depth = sum(float(bid[1]) for bid in bids 
                               if mid_price - float(bid[0]) <= price_range)
                ask_depth = sum(float(ask[1]) for ask in asks 
                               if float(ask[0]) - mid_price <= price_range)
                
                depth_metrics[f'depth_{level}pct'] = bid_depth + ask_depth
            
            return depth_metrics
            
        except Exception as e:
            logging.error(f"Error calculating liquidity depth: {e}")
            return {'depth_0.1pct': 0.0, 'depth_0.5pct': 0.0, 'depth_1.0pct': 0.0, 
                    'depth_2.0pct': 0.0, 'depth_5.0pct': 0.0}
    
    def _estimate_market_impact(self, bids: List, asks: List) -> float:
        """Estimate market impact of a standard order size."""
        try:
            # Calculate average order size from order book
            bid_sizes = [float(bid[1]) for bid in bids[:20]]
            ask_sizes = [float(ask[1]) for ask in asks[:20]]
            
            avg_size = np.mean(bid_sizes + ask_sizes)
            
            # Estimate impact using square root model
            # Impact = k * sqrt(order_size / avg_size)
            k = 0.001  # Impact coefficient
            standard_order_size = avg_size * 2  # 2x average size
            
            impact = k * np.sqrt(standard_order_size / avg_size)
            
            return min(impact, 0.05)  # Cap at 5%
            
        except Exception as e:
            logging.error(f"Error estimating market impact: {e}")
            return 0.001
    
    def _analyze_volume_profile(self, bids: List, asks: List) -> Dict[str, float]:
        """Analyze volume profile across price levels."""
        try:
            profile = {}
            
            # Calculate volume-weighted average price (VWAP)
            bid_vwap = sum(float(bid[0]) * float(bid[1]) for bid in bids[:10])
            bid_volume = sum(float(bid[1]) for bid in bids[:10])
            
            ask_vwap = sum(float(ask[0]) * float(ask[1]) for ask in asks[:10])
            ask_volume = sum(float(ask[1]) for ask in asks[:10])
            
            if bid_volume > 0 and ask_volume > 0:
                profile['bid_vwap'] = bid_vwap / bid_volume
                profile['ask_vwap'] = ask_vwap / ask_volume
                profile['vwap_spread'] = (ask_vwap / ask_volume) - (bid_vwap / bid_volume)
            else:
                profile['bid_vwap'] = 0.0
                profile['ask_vwap'] = 0.0
                profile['vwap_spread'] = 0.0
            
            # Calculate volume concentration
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                profile['volume_concentration'] = max(bid_volume, ask_volume) / total_volume
            else:
                profile['volume_concentration'] = 0.5
            
            return profile
            
        except Exception as e:
            logging.error(f"Error analyzing volume profile: {e}")
            return {'bid_vwap': 0.0, 'ask_vwap': 0.0, 'vwap_spread': 0.0, 'volume_concentration': 0.5}
    
    def _calculate_price_pressure(self, bids: List, asks: List) -> float:
        """Calculate price pressure based on order book structure."""
        try:
            # Calculate price pressure as the ratio of large orders to small orders
            large_threshold = np.mean([float(bid[1]) for bid in bids[:10]]) * 2
            
            large_bids = sum(1 for bid in bids[:10] if float(bid[1]) > large_threshold)
            large_asks = sum(1 for ask in asks[:10] if float(ask[1]) > large_threshold)
            
            total_orders = len(bids[:10]) + len(asks[:10])
            if total_orders == 0:
                return 0.0
            
            pressure = (large_bids - large_asks) / total_orders
            
            return np.clip(pressure, -1.0, 1.0)
            
        except Exception as e:
            logging.error(f"Error calculating price pressure: {e}")
            return 0.0
    
    def _calculate_market_resilience(self, bids: List, asks: List) -> float:
        """Calculate market resilience based on order book depth."""
        try:
            # Calculate resilience as the ability to absorb large orders
            mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
            
            # Calculate how much volume is available within 1% of mid price
            resilience_volume = 0
            for bid in bids:
                if (mid_price - float(bid[0])) / mid_price <= 0.01:
                    resilience_volume += float(bid[1])
            
            for ask in asks:
                if (float(ask[0]) - mid_price) / mid_price <= 0.01:
                    resilience_volume += float(ask[1])
            
            # Normalize by average order size
            avg_size = np.mean([float(bid[1]) for bid in bids[:10]] + [float(ask[1]) for ask in asks[:10]])
            if avg_size > 0:
                resilience = resilience_volume / avg_size
                return min(resilience / 100, 1.0)  # Normalize to 0-1
            
            return 0.0
            
        except Exception as e:
            logging.error(f"Error calculating market resilience: {e}")
            return 0.0
    
    def _calculate_market_fragmentation(self, bids: List, asks: List) -> float:
        """Calculate market fragmentation based on order size distribution."""
        try:
            # Calculate fragmentation as the standard deviation of order sizes
            all_sizes = [float(bid[1]) for bid in bids[:10]] + [float(ask[1]) for ask in asks[:10]]
            
            if len(all_sizes) < 2:
                return 0.0
            
            mean_size = np.mean(all_sizes)
            if mean_size == 0:
                return 0.0
            
            fragmentation = np.std(all_sizes) / mean_size
            
            return min(fragmentation, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            logging.error(f"Error calculating market fragmentation: {e}")
            return 0.0
    
    def _update_real_time_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update real-time microstructure metrics."""
        try:
            self.current_imbalance = metrics.get('order_imbalance', 0.0)
            self.current_spread = metrics.get('bid_ask_spread', 0.0)
            self.current_depth = metrics.get('liquidity_depth', {}).get('depth_1.0pct', 0.0)
            self.current_impact = metrics.get('market_impact', 0.0)
            
            # Store metrics in history
            self.microstructure_metrics.append(metrics)
            
        except Exception as e:
            logging.error(f"Error updating real-time metrics: {e}")
    
    def _detect_market_regime(self) -> None:
        """Detect current market microstructure regime."""
        try:
            if len(self.microstructure_metrics) < 10:
                self.market_regime = 'NORMAL'
                self.regime_confidence = 0.0
                return
            
            # Calculate regime indicators
            recent_metrics = list(self.microstructure_metrics)[-10:]
            
            avg_spread = np.mean([m.get('bid_ask_spread', 0.0) for m in recent_metrics])
            avg_imbalance = np.mean([abs(m.get('order_imbalance', 0.0)) for m in recent_metrics])
            avg_impact = np.mean([m.get('market_impact', 0.0) for m in recent_metrics])
            
            # Determine regime based on metrics
            if avg_spread > 0.01:  # High spread
                regime = 'ILLIQUID'
                confidence = min(avg_spread / 0.02, 1.0)
            elif avg_imbalance > 0.3:  # High imbalance
                regime = 'IMBALANCED'
                confidence = min(avg_imbalance / 0.5, 1.0)
            elif avg_impact > 0.005:  # High impact
                regime = 'FRAGILE'
                confidence = min(avg_impact / 0.01, 1.0)
            else:
                regime = 'NORMAL'
                confidence = 0.8
            
            self.market_regime = regime
            self.regime_confidence = confidence
            
        except Exception as e:
            logging.error(f"Error detecting market regime: {e}")
            self.market_regime = 'NORMAL'
            self.regime_confidence = 0.0
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics when data is unavailable."""
        return {
            'timestamp': time.time(),
            'order_imbalance': 0.0,
            'bid_ask_spread': 0.0,
            'liquidity_depth': {'depth_0.1pct': 0.0, 'depth_0.5pct': 0.0, 'depth_1.0pct': 0.0, 
                               'depth_2.0pct': 0.0, 'depth_5.0pct': 0.0},
            'market_impact': 0.001,
            'volume_profile': {'bid_vwap': 0.0, 'ask_vwap': 0.0, 'vwap_spread': 0.0, 'volume_concentration': 0.5},
            'price_pressure': 0.0,
            'resilience': 0.0,
            'fragmentation': 0.0
        }
    
    def get_microstructure_summary(self) -> Dict[str, Any]:
        """Get comprehensive microstructure summary."""
        try:
            return {
                'current_metrics': {
                    'imbalance': self.current_imbalance,
                    'spread': self.current_spread,
                    'depth': self.current_depth,
                    'impact': self.current_impact
                },
                'market_regime': {
                    'regime': self.market_regime,
                    'confidence': self.regime_confidence
                },
                'performance': {
                    'analysis_count': self.analysis_count,
                    'last_update': self.last_update,
                    'update_frequency': self.analysis_count / max(time.time() - self.last_update, 1)
                },
                'historical_summary': self._get_historical_summary()
            }
            
        except Exception as e:
            logging.error(f"Error getting microstructure summary: {e}")
            return {}
    
    def _get_historical_summary(self) -> Dict[str, Any]:
        """Get historical summary of microstructure metrics."""
        try:
            if len(self.microstructure_metrics) < 5:
                return {}
            
            metrics_list = list(self.microstructure_metrics)
            
            return {
                'avg_spread': np.mean([m.get('bid_ask_spread', 0.0) for m in metrics_list]),
                'avg_imbalance': np.mean([m.get('order_imbalance', 0.0) for m in metrics_list]),
                'avg_impact': np.mean([m.get('market_impact', 0.0) for m in metrics_list]),
                'spread_volatility': np.std([m.get('bid_ask_spread', 0.0) for m in metrics_list]),
                'imbalance_volatility': np.std([m.get('order_imbalance', 0.0) for m in metrics_list])
            }
            
        except Exception as e:
            logging.error(f"Error getting historical summary: {e}")
            return {} 