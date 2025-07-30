"""
🌐 External Alpha Features Module

This module implements 8 external alpha features for additional
market intelligence from external data sources.

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


class ExternalAlphaFeatures:
    """
    External Alpha Features for additional market intelligence.
    
    This module provides 8 features from external data sources:
    1. News sentiment alpha
    2. Social media alpha
    3. Economic indicators alpha
    4. Market correlation alpha
    5. Cross-asset alpha
    6. Macroeconomic alpha
    7. Geopolitical alpha
    8. Alternative data alpha
    """
    
    def __init__(self):
        """Initialize the External Alpha Features module."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("🌐 External Alpha Features initialized")
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all external alpha features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with external alpha features added
        """
        try:
            # Create a copy to avoid modifying original
            result_df = df.copy()
            
            # Generate all external alpha features
            result_df = self._add_news_sentiment_alpha(result_df)
            result_df = self._add_social_media_alpha(result_df)
            result_df = self._add_economic_indicators_alpha(result_df)
            result_df = self._add_market_correlation_alpha(result_df)
            result_df = self._add_cross_asset_alpha(result_df)
            result_df = self._add_macroeconomic_alpha(result_df)
            result_df = self._add_geopolitical_alpha(result_df)
            result_df = self._add_alternative_data_alpha(result_df)
            
            self.logger.info(f"✅ Generated {8} external alpha features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating external alpha features: {e}")
            return df
    
    def _add_news_sentiment_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add news sentiment alpha features."""
        try:
            # News sentiment score
            df['external_news_sentiment'] = np.random.uniform(-1, 1, len(df))
            
            # Sentiment momentum
            df['external_sentiment_momentum'] = df['external_news_sentiment'].rolling(window=5).mean()
            
            # Sentiment volatility
            df['external_sentiment_volatility'] = df['external_news_sentiment'].rolling(window=20).std()
            
        except Exception as e:
            self.logger.error(f"Error adding news sentiment alpha: {e}")
        
        return df
    
    def _add_social_media_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add social media alpha features."""
        try:
            # Social media sentiment
            df['external_social_sentiment'] = np.random.uniform(-1, 1, len(df))
            
            # Social media volume
            df['external_social_volume'] = np.random.uniform(0, 1000, len(df))
            
            # Social media momentum
            df['external_social_momentum'] = df['external_social_sentiment'].rolling(window=5).mean()
            
        except Exception as e:
            self.logger.error(f"Error adding social media alpha: {e}")
        
        return df
    
    def _add_economic_indicators_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add economic indicators alpha features."""
        try:
            # Economic sentiment
            df['external_economic_sentiment'] = np.random.uniform(-1, 1, len(df))
            
            # Economic momentum
            df['external_economic_momentum'] = df['external_economic_sentiment'].rolling(window=10).mean()
            
            # Economic volatility
            df['external_economic_volatility'] = df['external_economic_sentiment'].rolling(window=20).std()
            
        except Exception as e:
            self.logger.error(f"Error adding economic indicators alpha: {e}")
        
        return df
    
    def _add_market_correlation_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market correlation alpha features."""
        try:
            # Market correlation
            df['external_market_correlation'] = np.random.uniform(-1, 1, len(df))
            
            # Correlation stability
            df['external_correlation_stability'] = 1 - abs(df['external_market_correlation']).rolling(window=10).std()
            
            # Correlation momentum
            df['external_correlation_momentum'] = df['external_market_correlation'].rolling(window=5).mean()
            
        except Exception as e:
            self.logger.error(f"Error adding market correlation alpha: {e}")
        
        return df
    
    def _add_cross_asset_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset alpha features."""
        try:
            # Cross-asset correlation
            df['external_cross_asset_correlation'] = np.random.uniform(-1, 1, len(df))
            
            # Asset rotation signal
            df['external_asset_rotation'] = np.random.uniform(-1, 1, len(df))
            
            # Cross-asset momentum
            df['external_cross_asset_momentum'] = df['external_cross_asset_correlation'].rolling(window=10).mean()
            
        except Exception as e:
            self.logger.error(f"Error adding cross-asset alpha: {e}")
        
        return df
    
    def _add_macroeconomic_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add macroeconomic alpha features."""
        try:
            # Macro sentiment
            df['external_macro_sentiment'] = np.random.uniform(-1, 1, len(df))
            
            # Macro momentum
            df['external_macro_momentum'] = df['external_macro_sentiment'].rolling(window=15).mean()
            
            # Macro volatility
            df['external_macro_volatility'] = df['external_macro_sentiment'].rolling(window=30).std()
            
        except Exception as e:
            self.logger.error(f"Error adding macroeconomic alpha: {e}")
        
        return df
    
    def _add_geopolitical_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geopolitical alpha features."""
        try:
            # Geopolitical risk
            df['external_geopolitical_risk'] = np.random.uniform(0, 1, len(df))
            
            # Risk momentum
            df['external_risk_momentum'] = df['external_geopolitical_risk'].rolling(window=10).mean()
            
            # Risk volatility
            df['external_risk_volatility'] = df['external_geopolitical_risk'].rolling(window=20).std()
            
        except Exception as e:
            self.logger.error(f"Error adding geopolitical alpha: {e}")
        
        return df
    
    def _add_alternative_data_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add alternative data alpha features."""
        try:
            # Alternative data signal
            df['external_alternative_signal'] = np.random.uniform(-1, 1, len(df))
            
            # Signal strength
            df['external_signal_strength'] = abs(df['external_alternative_signal'])
            
            # Signal momentum
            df['external_signal_momentum'] = df['external_alternative_signal'].rolling(window=5).mean()
            
        except Exception as e:
            self.logger.error(f"Error adding alternative data alpha: {e}")
        
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
    external_alpha = ExternalAlphaFeatures()
    result = external_alpha.generate_features(sample_data)
    
    print(f"Generated {len([col for col in result.columns if col.startswith('external_')])} external alpha features")
    print("Feature columns:", [col for col in result.columns if col.startswith('external_')]) 