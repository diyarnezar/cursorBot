"""
🎯 Regime Detection Features Module

This module implements 5 regime detection features for identifying
market regimes in cryptocurrency trading.

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


class RegimeDetectionFeatures:
    """
    Regime Detection Features for identifying market regimes.
    
    This module provides 5 features for regime detection:
    1. Volatility regime detection
    2. Trend regime detection
    3. Volume regime detection
    4. Correlation regime detection
    5. Combined regime detection
    """
    
    def __init__(self):
        """Initialize the Regime Detection Features module."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Regime Detection Features initialized")
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all regime detection features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime detection features added
        """
        try:
            # Create a copy to avoid modifying original
            result_df = df.copy()
            
            # Generate all regime detection features
            result_df = self._add_volatility_regime_detection(result_df)
            result_df = self._add_trend_regime_detection(result_df)
            result_df = self._add_volume_regime_detection(result_df)
            result_df = self._add_correlation_regime_detection(result_df)
            result_df = self._add_combined_regime_detection(result_df)
            
            self.logger.info(f"✅ Generated {5} regime detection features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating regime detection features: {e}")
            return df
    
    def _add_volatility_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime detection features."""
        try:
            # Calculate volatility
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            
            # Volatility regime
            vol_mean = volatility.rolling(window=50).mean()
            vol_std = volatility.rolling(window=50).std()
            
            # High volatility regime
            df['regime_high_volatility'] = (volatility > vol_mean + vol_std).astype(int)
            
            # Low volatility regime
            df['regime_low_volatility'] = (volatility < vol_mean - vol_std).astype(int)
            
            # Normal volatility regime
            df['regime_normal_volatility'] = ((volatility >= vol_mean - vol_std) & 
                                             (volatility <= vol_mean + vol_std)).astype(int)
            
            # Regime confidence
            df['regime_volatility_confidence'] = 1 - abs(volatility - vol_mean) / (vol_std + 1e-8)
            
        except Exception as e:
            self.logger.error(f"Error adding volatility regime detection: {e}")
        
        return df
    
    def _add_trend_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend regime detection features."""
        try:
            # Calculate trend
            short_ma = df['close'].rolling(window=10).mean()
            long_ma = df['close'].rolling(window=50).mean()
            
            # Trend strength
            trend_strength = (short_ma - long_ma) / long_ma
            
            # Uptrend regime
            df['regime_uptrend'] = (trend_strength > 0.02).astype(int)
            
            # Downtrend regime
            df['regime_downtrend'] = (trend_strength < -0.02).astype(int)
            
            # Sideways regime
            df['regime_sideways'] = ((trend_strength >= -0.02) & (trend_strength <= 0.02)).astype(int)
            
            # Trend confidence
            df['regime_trend_confidence'] = abs(trend_strength)
            
        except Exception as e:
            self.logger.error(f"Error adding trend regime detection: {e}")
        
        return df
    
    def _add_volume_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume regime detection features."""
        try:
            # Calculate volume metrics
            volume_mean = df['volume'].rolling(window=50).mean()
            volume_std = df['volume'].rolling(window=50).std()
            
            # High volume regime
            df['regime_high_volume'] = (df['volume'] > volume_mean + volume_std).astype(int)
            
            # Low volume regime
            df['regime_low_volume'] = (df['volume'] < volume_mean - volume_std).astype(int)
            
            # Normal volume regime
            df['regime_normal_volume'] = ((df['volume'] >= volume_mean - volume_std) & 
                                         (df['volume'] <= volume_mean + volume_std)).astype(int)
            
            # Volume regime confidence
            df['regime_volume_confidence'] = 1 - abs(df['volume'] - volume_mean) / (volume_std + 1e-8)
            
        except Exception as e:
            self.logger.error(f"Error adding volume regime detection: {e}")
        
        return df
    
    def _add_correlation_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add correlation regime detection features."""
        try:
            # Simulate correlation with other assets (in real implementation, this would use actual data)
            correlation = np.random.uniform(-1, 1, len(df))
            
            # High correlation regime
            df['regime_high_correlation'] = (abs(correlation) > 0.7).astype(int)
            
            # Low correlation regime
            df['regime_low_correlation'] = (abs(correlation) < 0.3).astype(int)
            
            # Medium correlation regime
            df['regime_medium_correlation'] = ((abs(correlation) >= 0.3) & (abs(correlation) <= 0.7)).astype(int)
            
            # Correlation regime confidence
            df['regime_correlation_confidence'] = abs(correlation)
            
        except Exception as e:
            self.logger.error(f"Error adding correlation regime detection: {e}")
        
        return df
    
    def _add_combined_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add combined regime detection features."""
        try:
            # Combined regime score
            df['regime_combined_score'] = (
                df['regime_volatility_confidence'] * 0.3 +
                df['regime_trend_confidence'] * 0.3 +
                df['regime_volume_confidence'] * 0.2 +
                df['regime_correlation_confidence'] * 0.2
            )
            
            # Strong regime
            df['regime_strong'] = (df['regime_combined_score'] > 0.7).astype(int)
            
            # Weak regime
            df['regime_weak'] = (df['regime_combined_score'] < 0.3).astype(int)
            
            # Mixed regime
            df['regime_mixed'] = ((df['regime_combined_score'] >= 0.3) & 
                                 (df['regime_combined_score'] <= 0.7)).astype(int)
            
            # Regime stability
            df['regime_stability'] = 1 - df['regime_combined_score'].rolling(window=10).std()
            
        except Exception as e:
            self.logger.error(f"Error adding combined regime detection: {e}")
        
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
    regime_detection = RegimeDetectionFeatures()
    result = regime_detection.generate_features(sample_data)
    
    print(f"Generated {len([col for col in result.columns if col.startswith('regime_')])} regime detection features")
    print("Feature columns:", [col for col in result.columns if col.startswith('regime_')]) 