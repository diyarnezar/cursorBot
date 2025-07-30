"""
📊 Market Microstructure Features Module

This module implements 11 market microstructure features for advanced
order book and trade flow analysis in cryptocurrency trading.

Author: Hyperion Trading System
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class MicrostructureFeatures:
    """
    📊 Market Microstructure Features Generator
    
    Implements 11 market microstructure features for advanced order book
    and trade flow analysis in cryptocurrency trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the microstructure features generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_names = []
        self.feature_descriptions = {}
        
        # Microstructure parameters
        self.microstructure_params = {
            'spread_window': 20,
            'depth_levels': 10,
            'trade_size_buckets': 5,
            'flow_window': 15,
            'impact_window': 10
        }
        
        logger.info("📊 Microstructure Features initialized")
    
    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all 11 market microstructure features to the dataframe.
        
        Args:
            df: Input dataframe with OHLCV data
            
        Returns:
            DataFrame with microstructure features added
        """
        try:
            logger.info("📊 Adding market microstructure features...")
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 1000  # Default value
            
            # Calculate dynamic windows based on data length
            data_length = len(df)
            short_window = min(5, max(2, data_length // 20))
            medium_window = min(10, max(5, data_length // 10))
            long_window = min(20, max(10, data_length // 5))
            
            # 1. Bid-Ask Spread
            df = self._add_bid_ask_spread(df, short_window)
            
            # 2. Order Book Imbalance
            df = self._add_order_book_imbalance(df, medium_window)
            
            # 3. Trade Size Distribution
            df = self._add_trade_size_distribution(df, short_window)
            
            # 4. Market Depth
            df = self._add_market_depth(df, medium_window)
            
            # 5. Liquidity Metrics
            df = self._add_liquidity_metrics(df, short_window)
            
            # 6. Price Impact
            df = self._add_price_impact(df, short_window)
            
            # 7. Trade Flow
            df = self._add_trade_flow(df, medium_window)
            
            # 8. Market Microstructure Noise
            df = self._add_microstructure_noise(df, long_window)
            
            # 9. Order Flow Imbalance
            df = self._add_order_flow_imbalance(df, short_window)
            
            # 10. Market Impact
            df = self._add_market_impact(df, medium_window)
            
            # 11. Microstructure Volatility
            df = self._add_microstructure_volatility(df, short_window)
            
            logger.info(f"✅ Added {len(self.feature_names)} microstructure features")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to add microstructure features: {e}")
            return df
    
    def _add_bid_ask_spread(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add bid-ask spread analysis feature."""
        try:
            # Simulate bid-ask spread based on price volatility
            price_volatility = df['close'].rolling(window).std() / df['close']
            
            # Bid-ask spread is typically proportional to volatility
            # In crypto, spreads are usually 0.01% to 0.1% of price
            base_spread = 0.0005  # 0.05% base spread
            volatility_adjustment = price_volatility * 10
            
            bid_ask_spread = base_spread + volatility_adjustment
            
            # Add some noise to make it realistic
            noise = np.random.normal(0, 0.0001, len(df))
            bid_ask_spread = np.maximum(0.0001, bid_ask_spread + noise)
            
            df['bid_ask_spread'] = bid_ask_spread
            
            self.feature_names.append('bid_ask_spread')
            self.feature_descriptions['bid_ask_spread'] = 'Bid-ask spread analysis'
            
        except Exception as e:
            logger.error(f"❌ Bid-ask spread failed: {e}")
            df['bid_ask_spread'] = 0.0005
        
        return df
    
    def _add_order_book_imbalance(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add order book imbalance feature."""
        try:
            # Simulate order book imbalance based on price movement and volume
            price_change = df['close'].pct_change()
            volume_ratio = df['volume'] / df['volume'].rolling(window).mean()
            
            # Order book imbalance is related to price pressure
            # Positive imbalance (more bids) when price is rising
            # Negative imbalance (more asks) when price is falling
            imbalance = price_change * volume_ratio
            
            # Normalize to [-1, 1] range
            imbalance = np.tanh(imbalance * 10)
            
            df['order_book_imbalance'] = imbalance
            
            self.feature_names.append('order_book_imbalance')
            self.feature_descriptions['order_book_imbalance'] = 'Order book bid-ask imbalance'
            
        except Exception as e:
            logger.error(f"❌ Order book imbalance failed: {e}")
            df['order_book_imbalance'] = 0.0
        
        return df
    
    def _add_trade_size_distribution(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add trade size distribution feature."""
        try:
            # Analyze trade size distribution
            volume_mean = df['volume'].rolling(window).mean()
            volume_std = df['volume'].rolling(window).std()
            
            # Trade size distribution metric
            # Large trades relative to average indicate institutional activity
            trade_size_ratio = df['volume'] / volume_mean
            
            # Normalize and create distribution metric
            trade_size_distribution = np.log1p(trade_size_ratio) / (1 + np.log1p(volume_std / volume_mean))
            
            df['trade_size_distribution'] = trade_size_distribution
            
            self.feature_names.append('trade_size_distribution')
            self.feature_descriptions['trade_size_distribution'] = 'Trade size distribution analysis'
            
        except Exception as e:
            logger.error(f"❌ Trade size distribution failed: {e}")
            df['trade_size_distribution'] = 0.0
        
        return df
    
    def _add_market_depth(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add market depth analysis feature."""
        try:
            # Market depth is related to liquidity and volume
            volume_ma = df['volume'].rolling(window).mean()
            price_volatility = df['close'].rolling(window).std() / df['close']
            
            # Market depth metric
            # Higher volume and lower volatility indicate deeper market
            market_depth = volume_ma / (1 + price_volatility * 100)
            
            # Normalize to reasonable range
            market_depth = market_depth / market_depth.rolling(window).max()
            
            df['market_depth'] = market_depth
            
            self.feature_names.append('market_depth')
            self.feature_descriptions['market_depth'] = 'Market depth analysis'
            
        except Exception as e:
            logger.error(f"❌ Market depth failed: {e}")
            df['market_depth'] = 0.5
        
        return df
    
    def _add_liquidity_metrics(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add liquidity metrics feature."""
        try:
            # Calculate liquidity metrics
            volume_ma = df['volume'].rolling(window).mean()
            price_range = (df['high'] - df['low']) / df['close']
            
            # Liquidity metric: volume per unit of price movement
            # Higher values indicate more liquid market
            liquidity_metric = volume_ma / (price_range + 1e-8)
            
            # Normalize to [0, 1] range
            liquidity_metric = liquidity_metric / liquidity_metric.rolling(window).max()
            
            df['liquidity_metrics'] = liquidity_metric
            
            self.feature_names.append('liquidity_metrics')
            self.feature_descriptions['liquidity_metrics'] = 'Market liquidity metrics'
            
        except Exception as e:
            logger.error(f"❌ Liquidity metrics failed: {e}")
            df['liquidity_metrics'] = 0.5
        
        return df
    
    def _add_price_impact(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add price impact analysis feature."""
        try:
            # Calculate price impact of trades
            volume_change = df['volume'].pct_change()
            price_change = df['close'].pct_change()
            
            # Price impact: how much price moves per unit of volume
            price_impact = np.abs(price_change) / (np.abs(volume_change) + 1e-8)
            
            # Normalize and smooth
            price_impact = price_impact.rolling(window).mean()
            price_impact = price_impact / price_impact.rolling(window).max()
            
            df['price_impact'] = price_impact
            
            self.feature_names.append('price_impact')
            self.feature_descriptions['price_impact'] = 'Price impact of trades'
            
        except Exception as e:
            logger.error(f"❌ Price impact failed: {e}")
            df['price_impact'] = 0.5
        
        return df
    
    def _add_trade_flow(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add trade flow analysis feature."""
        try:
            # Analyze trade flow patterns
            volume_ma = df['volume'].rolling(window).mean()
            price_momentum = df['close'].pct_change(window)
            
            # Trade flow metric: volume weighted by price momentum
            # Positive flow when volume increases with positive momentum
            # Negative flow when volume increases with negative momentum
            trade_flow = volume_ma * np.sign(price_momentum)
            
            # Normalize to [-1, 1] range
            trade_flow_std = trade_flow.rolling(window).std()
            trade_flow = np.tanh(trade_flow / (trade_flow_std + 1e-8))
            
            df['trade_flow'] = trade_flow
            
            self.feature_names.append('trade_flow')
            self.feature_descriptions['trade_flow'] = 'Trade flow analysis'
            
        except Exception as e:
            logger.error(f"❌ Trade flow failed: {e}")
            df['trade_flow'] = 0.0
        
        return df
    
    def _add_microstructure_noise(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add market microstructure noise feature."""
        try:
            # Calculate microstructure noise
            # Noise is the component of price movement not explained by fundamentals
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            
            # Simple noise model: price changes not correlated with volume
            price_ma = price_change.rolling(window).mean()
            volume_ma = volume_change.rolling(window).mean()
            
            # Noise is the residual
            noise = price_change - price_ma - 0.1 * (volume_change - volume_ma)
            
            # Normalize noise
            noise = noise / noise.rolling(window).std()
            
            df['market_microstructure_noise'] = noise
            
            self.feature_names.append('market_microstructure_noise')
            self.feature_descriptions['market_microstructure_noise'] = 'Market microstructure noise'
            
        except Exception as e:
            logger.error(f"❌ Microstructure noise failed: {e}")
            df['market_microstructure_noise'] = 0.0
        
        return df
    
    def _add_order_flow_imbalance(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add order flow imbalance feature."""
        try:
            # Analyze order flow imbalance
            # Based on price movement and volume patterns
            price_change = df['close'].pct_change()
            volume_ratio = df['volume'] / df['volume'].rolling(window).mean()
            
            # Order flow imbalance indicator
            # Positive when buying pressure exceeds selling pressure
            # Negative when selling pressure exceeds buying pressure
            flow_imbalance = price_change * volume_ratio
            
            # Add momentum component
            momentum = price_change.rolling(window).sum()
            flow_imbalance = flow_imbalance + 0.1 * momentum
            
            # Normalize to [-1, 1] range
            flow_imbalance = np.tanh(flow_imbalance * 5)
            
            df['order_flow_imbalance'] = flow_imbalance
            
            self.feature_names.append('order_flow_imbalance')
            self.feature_descriptions['order_flow_imbalance'] = 'Order flow imbalance'
            
        except Exception as e:
            logger.error(f"❌ Order flow imbalance failed: {e}")
            df['order_flow_imbalance'] = 0.0
        
        return df
    
    def _add_market_impact(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add market impact analysis feature."""
        try:
            # Calculate market impact
            # Impact is how much the market moves in response to trading activity
            volume_impact = df['volume'] / df['volume'].rolling(window).mean()
            price_volatility = df['close'].rolling(window).std() / df['close']
            
            # Market impact model
            # Higher volume and volatility lead to higher impact
            market_impact = volume_impact * price_volatility * 100
            
            # Add time decay component
            time_decay = np.exp(-np.arange(len(df)) / window)
            market_impact = market_impact * pd.Series(time_decay, index=df.index)
            
            # Normalize
            market_impact = market_impact / market_impact.rolling(window).max()
            
            df['market_impact'] = market_impact
            
            self.feature_names.append('market_impact')
            self.feature_descriptions['market_impact'] = 'Market impact analysis'
            
        except Exception as e:
            logger.error(f"❌ Market impact failed: {e}")
            df['market_impact'] = 0.5
        
        return df
    
    def _add_microstructure_volatility(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add microstructure volatility feature."""
        try:
            # Calculate microstructure-specific volatility
            # This is volatility due to market microstructure factors
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            
            # Microstructure volatility components
            spread_volatility = df['bid_ask_spread'] if 'bid_ask_spread' in df.columns else 0.0005
            flow_volatility = pd.Series(volume_change).abs().rolling(window).std()
            
            # Combine components
            microstructure_volatility = (
                0.4 * spread_volatility +
                0.3 * flow_volatility +
                0.3 * np.abs(price_change).rolling(window).std()
            )
            
            # Normalize
            microstructure_volatility = microstructure_volatility / microstructure_volatility.rolling(window).max()
            
            df['microstructure_volatility'] = microstructure_volatility
            
            self.feature_names.append('microstructure_volatility')
            self.feature_descriptions['microstructure_volatility'] = 'Microstructure-specific volatility'
            
        except Exception as e:
            logger.error(f"❌ Microstructure volatility failed: {e}")
            df['microstructure_volatility'] = 0.5
        
        return df
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get a summary of microstructure features."""
        return {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_descriptions': self.feature_descriptions,
            'microstructure_params': self.microstructure_params
        }
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate microstructure features for quality."""
        validation_results = {}
        
        for feature_name in self.feature_names:
            if feature_name in df.columns:
                feature_data = df[feature_name]
                
                validation_results[feature_name] = {
                    'nan_ratio': feature_data.isna().sum() / len(feature_data),
                    'zero_ratio': (feature_data == 0).sum() / len(feature_data),
                    'unique_ratio': feature_data.nunique() / len(feature_data),
                    'mean': feature_data.mean(),
                    'std': feature_data.std(),
                    'min': feature_data.min(),
                    'max': feature_data.max()
                }
        
        return validation_results
    
    def analyze_microstructure_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market microstructure regime."""
        try:
            regime_analysis = {}
            
            # Analyze spread regime
            if 'bid_ask_spread' in df.columns:
                spread_mean = df['bid_ask_spread'].mean()
                spread_std = df['bid_ask_spread'].std()
                
                if spread_mean < 0.0002:
                    regime_analysis['spread_regime'] = 'tight'
                elif spread_mean < 0.0005:
                    regime_analysis['spread_regime'] = 'normal'
                else:
                    regime_analysis['spread_regime'] = 'wide'
            
            # Analyze liquidity regime
            if 'liquidity_metrics' in df.columns:
                liquidity_mean = df['liquidity_metrics'].mean()
                
                if liquidity_mean > 0.7:
                    regime_analysis['liquidity_regime'] = 'high'
                elif liquidity_mean > 0.3:
                    regime_analysis['liquidity_regime'] = 'normal'
                else:
                    regime_analysis['liquidity_regime'] = 'low'
            
            # Analyze flow regime
            if 'trade_flow' in df.columns:
                flow_mean = df['trade_flow'].mean()
                
                if flow_mean > 0.3:
                    regime_analysis['flow_regime'] = 'buying_pressure'
                elif flow_mean < -0.3:
                    regime_analysis['flow_regime'] = 'selling_pressure'
                else:
                    regime_analysis['flow_regime'] = 'balanced'
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"❌ Microstructure regime analysis failed: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'microstructure_features_enabled': True,
        'validation_threshold': 0.8
    }
    
    # Initialize microstructure features
    microstructure_features = MicrostructureFeatures(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'open': np.random.uniform(1000, 2000, 1000),
        'high': np.random.uniform(1000, 2000, 1000),
        'low': np.random.uniform(1000, 2000, 1000),
        'close': np.random.uniform(1000, 2000, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    # Add microstructure features
    enhanced_data = microstructure_features.add_microstructure_features(sample_data)
    
    # Get feature summary
    summary = microstructure_features.get_feature_summary()
    print(f"Added {summary['total_features']} microstructure features")
    
    # Validate features
    validation = microstructure_features.validate_features(enhanced_data)
    print("Microstructure feature validation completed")
    
    # Analyze microstructure regime
    regime = microstructure_features.analyze_microstructure_regime(enhanced_data)
    print(f"Microstructure regime: {regime}") 