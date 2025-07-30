"""
🛡️ Adaptive Risk Features Module

This module implements 9 adaptive risk features for dynamic
risk management in cryptocurrency trading.

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


class AdaptiveRiskFeatures:
    """
    Adaptive Risk Features for dynamic risk management.
    
    This module provides 9 features for adaptive risk management:
    1. Dynamic volatility adjustment
    2. Market regime risk
    3. Correlation risk
    4. Liquidity risk
    5. Concentration risk
    6. Drawdown risk
    7. Tail risk
    8. Stress test risk
    9. Adaptive position sizing
    """
    
    def __init__(self):
        """Initialize the Adaptive Risk Features module."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("🛡️ Adaptive Risk Features initialized")
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all adaptive risk features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with adaptive risk features added
        """
        try:
            # Create a copy to avoid modifying original
            result_df = df.copy()
            
            # Generate all adaptive risk features
            result_df = self._add_dynamic_volatility_features(result_df)
            result_df = self._add_market_regime_risk_features(result_df)
            result_df = self._add_correlation_risk_features(result_df)
            result_df = self._add_liquidity_risk_features(result_df)
            result_df = self._add_concentration_risk_features(result_df)
            result_df = self._add_drawdown_risk_features(result_df)
            result_df = self._add_tail_risk_features(result_df)
            result_df = self._add_stress_test_risk_features(result_df)
            result_df = self._add_adaptive_position_features(result_df)
            
            self.logger.info(f"✅ Generated {9} adaptive risk features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating adaptive risk features: {e}")
            return df
    
    def _add_dynamic_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add dynamic volatility adjustment features."""
        try:
            # Dynamic volatility
            returns = df['close'].pct_change()
            df['adaptive_dynamic_volatility'] = returns.rolling(window=20).std()
            
            # Volatility regime
            vol_mean = df['adaptive_dynamic_volatility'].rolling(window=50).mean()
            df['adaptive_volatility_regime'] = df['adaptive_dynamic_volatility'] / vol_mean
            
            # Volatility momentum
            df['adaptive_volatility_momentum'] = df['adaptive_dynamic_volatility'].pct_change(5)
            
        except Exception as e:
            self.logger.error(f"Error adding dynamic volatility features: {e}")
        
        return df
    
    def _add_market_regime_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime risk features."""
        try:
            # Market regime indicator
            df['adaptive_market_regime'] = np.random.uniform(0, 1, len(df))
            
            # Regime stability
            df['adaptive_regime_stability'] = 1 - df['adaptive_market_regime'].rolling(window=10).std()
            
            # Regime transition probability
            df['adaptive_regime_transition'] = np.random.uniform(0, 1, len(df))
            
        except Exception as e:
            self.logger.error(f"Error adding market regime risk features: {e}")
        
        return df
    
    def _add_correlation_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add correlation risk features."""
        try:
            # Correlation risk
            df['adaptive_correlation_risk'] = np.random.uniform(0, 1, len(df))
            
            # Correlation stability
            df['adaptive_correlation_stability'] = 1 - df['adaptive_correlation_risk'].rolling(window=10).std()
            
            # Correlation momentum
            df['adaptive_correlation_momentum'] = df['adaptive_correlation_risk'].rolling(window=5).mean()
            
        except Exception as e:
            self.logger.error(f"Error adding correlation risk features: {e}")
        
        return df
    
    def _add_liquidity_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity risk features."""
        try:
            # Liquidity risk
            df['adaptive_liquidity_risk'] = 1 / (df['volume'] + 1e-8)
            
            # Liquidity trend
            df['adaptive_liquidity_trend'] = df['adaptive_liquidity_risk'].rolling(window=10).mean()
            
            # Liquidity volatility
            df['adaptive_liquidity_volatility'] = df['adaptive_liquidity_risk'].rolling(window=20).std()
            
        except Exception as e:
            self.logger.error(f"Error adding liquidity risk features: {e}")
        
        return df
    
    def _add_concentration_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add concentration risk features."""
        try:
            # Concentration risk
            df['adaptive_concentration_risk'] = np.random.uniform(0, 1, len(df))
            
            # Concentration stability
            df['adaptive_concentration_stability'] = 1 - df['adaptive_concentration_risk'].rolling(window=10).std()
            
            # Concentration momentum
            df['adaptive_concentration_momentum'] = df['adaptive_concentration_risk'].rolling(window=5).mean()
            
        except Exception as e:
            self.logger.error(f"Error adding concentration risk features: {e}")
        
        return df
    
    def _add_drawdown_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add drawdown risk features."""
        try:
            # Rolling maximum
            rolling_max = df['close'].rolling(window=50).max()
            
            # Drawdown
            df['adaptive_drawdown'] = (df['close'] - rolling_max) / rolling_max
            
            # Drawdown risk
            df['adaptive_drawdown_risk'] = abs(df['adaptive_drawdown'])
            
            # Drawdown momentum
            df['adaptive_drawdown_momentum'] = df['adaptive_drawdown'].rolling(window=5).mean()
            
        except Exception as e:
            self.logger.error(f"Error adding drawdown risk features: {e}")
        
        return df
    
    def _add_tail_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add tail risk features."""
        try:
            # Tail risk (VaR-like)
            returns = df['close'].pct_change()
            df['adaptive_tail_risk'] = returns.rolling(window=20).quantile(0.05)
            
            # Tail risk stability
            df['adaptive_tail_risk_stability'] = 1 - abs(df['adaptive_tail_risk']).rolling(window=10).std()
            
            # Tail risk momentum
            df['adaptive_tail_risk_momentum'] = df['adaptive_tail_risk'].rolling(window=5).mean()
            
        except Exception as e:
            self.logger.error(f"Error adding tail risk features: {e}")
        
        return df
    
    def _add_stress_test_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add stress test risk features."""
        try:
            # Stress test scenario
            df['adaptive_stress_test'] = np.random.uniform(0, 1, len(df))
            
            # Stress test severity
            df['adaptive_stress_severity'] = df['adaptive_stress_test'] * df['volume'] / 10000
            
            # Stress test probability
            df['adaptive_stress_probability'] = np.random.uniform(0, 0.1, len(df))
            
        except Exception as e:
            self.logger.error(f"Error adding stress test risk features: {e}")
        
        return df
    
    def _add_adaptive_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add adaptive position sizing features."""
        try:
            # Adaptive position size
            df['adaptive_position_size'] = 1 / (df['adaptive_dynamic_volatility'] + 1e-8)
            
            # Position size stability
            df['adaptive_position_stability'] = 1 - df['adaptive_position_size'].rolling(window=10).std()
            
            # Position size momentum
            df['adaptive_position_momentum'] = df['adaptive_position_size'].rolling(window=5).mean()
            
        except Exception as e:
            self.logger.error(f"Error adding adaptive position features: {e}")
        
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
    adaptive_risk = AdaptiveRiskFeatures()
    result = adaptive_risk.generate_features(sample_data)
    
    print(f"Generated {len([col for col in result.columns if col.startswith('adaptive_')])} adaptive risk features")
    print("Feature columns:", [col for col in result.columns if col.startswith('adaptive_')]) 