"""
📊 Volatility & Momentum Features Module

This module implements 9 volatility and momentum features for
advanced market analysis in cryptocurrency trading.

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


class VolatilityMomentumFeatures:
    """
    Volatility & Momentum Features for advanced market analysis.
    
    This module provides 9 features for volatility and momentum analysis:
    1. Volatility metrics
    2. Momentum indicators
    3. Volatility regimes
    4. Momentum regimes
    5. Volatility clustering
    6. Momentum persistence
    7. Volatility forecasting
    8. Momentum forecasting
    9. Combined signals
    """
    
    def __init__(self):
        """Initialize the Volatility & Momentum Features module."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("📊 Volatility & Momentum Features initialized")
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all volatility and momentum features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility and momentum features added
        """
        try:
            # Create a copy to avoid modifying original
            result_df = df.copy()
            
            # Generate all volatility and momentum features
            result_df = self._add_volatility_metrics(result_df)
            result_df = self._add_momentum_indicators(result_df)
            result_df = self._add_volatility_regimes(result_df)
            result_df = self._add_momentum_regimes(result_df)
            result_df = self._add_volatility_clustering(result_df)
            result_df = self._add_momentum_persistence(result_df)
            result_df = self._add_volatility_forecasting(result_df)
            result_df = self._add_momentum_forecasting(result_df)
            result_df = self._add_combined_signals(result_df)
            
            self.logger.info(f"✅ Generated {9} volatility and momentum features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating volatility and momentum features: {e}")
            return df
    
    def _add_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility metrics features."""
        try:
            # Price volatility
            returns = df['close'].pct_change()
            df['vol_mom_price_volatility'] = returns.rolling(window=20).std()
            
            # Volume volatility
            df['vol_mom_volume_volatility'] = df['volume'].pct_change().rolling(window=20).std()
            
            # High-low volatility
            df['vol_mom_hl_volatility'] = ((df['high'] - df['low']) / df['close']).rolling(window=20).std()
            
            # Volatility of volatility
            df['vol_mom_vol_of_vol'] = df['vol_mom_price_volatility'].rolling(window=10).std()
            
        except Exception as e:
            self.logger.error(f"Error adding volatility metrics: {e}")
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators features."""
        try:
            # Price momentum
            df['vol_mom_price_momentum'] = df['close'].pct_change(5)
            
            # Volume momentum
            df['vol_mom_volume_momentum'] = df['volume'].pct_change(5)
            
            # Momentum strength
            df['vol_mom_momentum_strength'] = abs(df['vol_mom_price_momentum'])
            
            # Momentum acceleration
            df['vol_mom_momentum_acceleration'] = df['vol_mom_price_momentum'].diff()
            
        except Exception as e:
            self.logger.error(f"Error adding momentum indicators: {e}")
        
        return df
    
    def _add_volatility_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime features."""
        try:
            # Volatility regime
            vol_mean = df['vol_mom_price_volatility'].rolling(window=50).mean()
            df['vol_mom_volatility_regime'] = df['vol_mom_price_volatility'] / vol_mean
            
            # High volatility regime
            df['vol_mom_high_vol_regime'] = (df['vol_mom_volatility_regime'] > 1.5).astype(int)
            
            # Low volatility regime
            df['vol_mom_low_vol_regime'] = (df['vol_mom_volatility_regime'] < 0.5).astype(int)
            
            # Regime stability
            df['vol_mom_vol_regime_stability'] = 1 - df['vol_mom_volatility_regime'].rolling(window=10).std()
            
        except Exception as e:
            self.logger.error(f"Error adding volatility regimes: {e}")
        
        return df
    
    def _add_momentum_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum regime features."""
        try:
            # Momentum regime
            mom_mean = df['vol_mom_price_momentum'].rolling(window=50).mean()
            df['vol_mom_momentum_regime'] = df['vol_mom_price_momentum'] / (mom_mean + 1e-8)
            
            # Strong momentum regime
            df['vol_mom_strong_mom_regime'] = (df['vol_mom_momentum_regime'] > 1.5).astype(int)
            
            # Weak momentum regime
            df['vol_mom_weak_mom_regime'] = (df['vol_mom_momentum_regime'] < 0.5).astype(int)
            
            # Regime stability
            df['vol_mom_mom_regime_stability'] = 1 - df['vol_mom_momentum_regime'].rolling(window=10).std()
            
        except Exception as e:
            self.logger.error(f"Error adding momentum regimes: {e}")
        
        return df
    
    def _add_volatility_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility clustering features."""
        try:
            # Volatility clustering
            df['vol_mom_vol_clustering'] = df['vol_mom_price_volatility'].rolling(window=5).mean()
            
            # Clustering strength
            df['vol_mom_vol_clustering_strength'] = df['vol_mom_vol_clustering'] / (df['vol_mom_price_volatility'] + 1e-8)
            
            # Clustering persistence
            df['vol_mom_vol_clustering_persistence'] = df['vol_mom_vol_clustering'].rolling(window=10).std()
            
        except Exception as e:
            self.logger.error(f"Error adding volatility clustering: {e}")
        
        return df
    
    def _add_momentum_persistence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum persistence features."""
        try:
            # Momentum persistence
            df['vol_mom_momentum_persistence'] = df['vol_mom_price_momentum'].rolling(window=5).mean()
            
            # Persistence strength
            df['vol_mom_momentum_persistence_strength'] = df['vol_mom_momentum_persistence'] / (abs(df['vol_mom_price_momentum']) + 1e-8)
            
            # Persistence stability
            df['vol_mom_momentum_persistence_stability'] = 1 - df['vol_mom_momentum_persistence'].rolling(window=10).std()
            
        except Exception as e:
            self.logger.error(f"Error adding momentum persistence: {e}")
        
        return df
    
    def _add_volatility_forecasting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility forecasting features."""
        try:
            # Volatility forecast
            df['vol_mom_volatility_forecast'] = df['vol_mom_price_volatility'].rolling(window=10).mean()
            
            # Forecast accuracy
            df['vol_mom_vol_forecast_accuracy'] = 1 - abs(df['vol_mom_price_volatility'] - df['vol_mom_volatility_forecast']) / (df['vol_mom_price_volatility'] + 1e-8)
            
            # Forecast trend
            df['vol_mom_vol_forecast_trend'] = df['vol_mom_volatility_forecast'].pct_change()
            
        except Exception as e:
            self.logger.error(f"Error adding volatility forecasting: {e}")
        
        return df
    
    def _add_momentum_forecasting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum forecasting features."""
        try:
            # Momentum forecast
            df['vol_mom_momentum_forecast'] = df['vol_mom_price_momentum'].rolling(window=10).mean()
            
            # Forecast accuracy
            df['vol_mom_mom_forecast_accuracy'] = 1 - abs(df['vol_mom_price_momentum'] - df['vol_mom_momentum_forecast']) / (abs(df['vol_mom_price_momentum']) + 1e-8)
            
            # Forecast trend
            df['vol_mom_mom_forecast_trend'] = df['vol_mom_momentum_forecast'].pct_change()
            
        except Exception as e:
            self.logger.error(f"Error adding momentum forecasting: {e}")
        
        return df
    
    def _add_combined_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add combined volatility and momentum signals."""
        try:
            # Combined signal
            df['vol_mom_combined_signal'] = df['vol_mom_price_momentum'] * (1 / (df['vol_mom_price_volatility'] + 1e-8))
            
            # Signal strength
            df['vol_mom_signal_strength'] = abs(df['vol_mom_combined_signal'])
            
            # Signal stability
            df['vol_mom_signal_stability'] = 1 - df['vol_mom_combined_signal'].rolling(window=10).std()
            
            # Signal momentum
            df['vol_mom_signal_momentum'] = df['vol_mom_combined_signal'].rolling(window=5).mean()
            
        except Exception as e:
            self.logger.error(f"Error adding combined signals: {e}")
        
        return df


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    # Initialize and generate features
    vol_mom = VolatilityMomentumFeatures()
    result = vol_mom.generate_features(sample_data)
    
    print(f"Generated {len([col for col in result.columns if col.startswith('vol_mom_')])} volatility and momentum features")
    print("Feature columns:", [col for col in result.columns if col.startswith('vol_mom_')]) 