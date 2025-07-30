"""
🧠 Psychology Features Module

This module implements 7 psychology features for market sentiment and
crowd behavior analysis in cryptocurrency trading.

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

class PsychologyFeatures:
    """
    🧠 Psychology Features Generator
    
    Implements 7 psychology features for market sentiment and crowd
    behavior analysis in cryptocurrency trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the psychology features generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_names = []
        self.feature_descriptions = {}
        
        # Psychology parameters
        self.psychology_params = {
            'fomo_threshold': 0.02,  # 2% price increase
            'panic_threshold': -0.02,  # 2% price decrease
            'euphoria_threshold': 0.05,  # 5% price increase
            'capitulation_threshold': -0.05,  # 5% price decrease
            'sentiment_window': 20,
            'crowd_window': 15
        }
        
        logger.info("🧠 Psychology Features initialized")
    
    def add_psychology_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all 7 psychology features to the dataframe.
        
        Args:
            df: Input dataframe with OHLCV data
            
        Returns:
            DataFrame with psychology features added
        """
        try:
            logger.info("🧠 Adding psychology features...")
            
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
            
            # 1. FOMO Indicator
            df = self._add_fomo_indicator(df, short_window)
            
            # 2. Panic Indicator
            df = self._add_panic_indicator(df, short_window)
            
            # 3. Euphoria Indicator
            df = self._add_euphoria_indicator(df, medium_window)
            
            # 4. Capitulation Indicator
            df = self._add_capitulation_indicator(df, medium_window)
            
            # 5. Greed/Fear Balance
            df = self._add_greed_fear_balance(df, long_window)
            
            # 6. Crowd Sentiment
            df = self._add_crowd_sentiment(df, medium_window)
            
            # 7. Herd Behavior
            df = self._add_herd_behavior(df, long_window)
            
            logger.info(f"✅ Added {len(self.feature_names)} psychology features")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to add psychology features: {e}")
            return df
    
    def _add_fomo_indicator(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add FOMO (Fear of Missing Out) indicator."""
        try:
            # Calculate price momentum and volume surge
            price_momentum = df['close'].pct_change(window)
            volume_surge = df['volume'] / df['volume'].rolling(window).mean()
            
            # FOMO conditions:
            # 1. Strong positive price momentum
            # 2. High volume relative to average
            # 3. Accelerating price movement
            price_acceleration = price_momentum.diff()
            
            # FOMO score
            fomo_score = (
                0.4 * np.tanh(price_momentum * 10) +  # Price momentum component
                0.3 * np.tanh(volume_surge - 1) +     # Volume surge component
                0.3 * np.tanh(price_acceleration * 20)  # Acceleration component
            )
            
            # Normalize to [0, 1] range
            fomo_score = (fomo_score + 1) / 2
            
            df['fomo_indicator'] = fomo_score
            
            self.feature_names.append('fomo_indicator')
            self.feature_descriptions['fomo_indicator'] = 'Fear of Missing Out indicator'
            
        except Exception as e:
            logger.error(f"❌ FOMO indicator failed: {e}")
            df['fomo_indicator'] = 0.5
        
        return df
    
    def _add_panic_indicator(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add panic indicator."""
        try:
            # Calculate negative price momentum and volume spike
            price_momentum = df['close'].pct_change(window)
            volume_spike = df['volume'] / df['volume'].rolling(window).mean()
            
            # Panic conditions:
            # 1. Strong negative price momentum
            # 2. High volume (panic selling)
            # 3. Accelerating downward movement
            price_acceleration = price_momentum.diff()
            
            # Panic score (negative momentum = positive panic)
            panic_score = (
                0.4 * np.tanh(-price_momentum * 10) +  # Negative momentum component
                0.3 * np.tanh(volume_spike - 1) +      # Volume spike component
                0.3 * np.tanh(-price_acceleration * 20)  # Negative acceleration component
            )
            
            # Normalize to [0, 1] range
            panic_score = (panic_score + 1) / 2
            
            df['panic_indicator'] = panic_score
            
            self.feature_names.append('panic_indicator')
            self.feature_descriptions['panic_indicator'] = 'Panic selling indicator'
            
        except Exception as e:
            logger.error(f"❌ Panic indicator failed: {e}")
            df['panic_indicator'] = 0.5
        
        return df
    
    def _add_euphoria_indicator(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add euphoria indicator."""
        try:
            # Calculate sustained positive momentum and extreme volume
            price_momentum = df['close'].pct_change(window)
            volume_extreme = df['volume'] / df['volume'].rolling(window * 2).mean()
            
            # Euphoria conditions:
            # 1. Sustained strong positive momentum
            # 2. Extreme volume (euphoric buying)
            # 3. Overbought conditions
            momentum_sustainability = price_momentum.rolling(window).mean()
            overbought_condition = (df['close'] - df['close'].rolling(window).min()) / (df['close'].rolling(window).max() - df['close'].rolling(window).min())
            
            # Euphoria score
            euphoria_score = (
                0.3 * np.tanh(price_momentum * 8) +           # Current momentum
                0.3 * np.tanh(momentum_sustainability * 8) +   # Sustained momentum
                0.2 * np.tanh(volume_extreme - 1.5) +         # Extreme volume
                0.2 * overbought_condition                     # Overbought condition
            )
            
            # Normalize to [0, 1] range
            euphoria_score = (euphoria_score + 1) / 2
            
            df['euphoria_indicator'] = euphoria_score
            
            self.feature_names.append('euphoria_indicator')
            self.feature_descriptions['euphoria_indicator'] = 'Euphoric buying indicator'
            
        except Exception as e:
            logger.error(f"❌ Euphoria indicator failed: {e}")
            df['euphoria_indicator'] = 0.5
        
        return df
    
    def _add_capitulation_indicator(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add capitulation indicator."""
        try:
            # Calculate sustained negative momentum and extreme volume
            price_momentum = df['close'].pct_change(window)
            volume_extreme = df['volume'] / df['volume'].rolling(window * 2).mean()
            
            # Capitulation conditions:
            # 1. Sustained strong negative momentum
            # 2. Extreme volume (capitulation selling)
            # 3. Oversold conditions
            momentum_sustainability = price_momentum.rolling(window).mean()
            oversold_condition = 1 - (df['close'] - df['close'].rolling(window).min()) / (df['close'].rolling(window).max() - df['close'].rolling(window).min())
            
            # Capitulation score (negative momentum = positive capitulation)
            capitulation_score = (
                0.3 * np.tanh(-price_momentum * 8) +           # Current negative momentum
                0.3 * np.tanh(-momentum_sustainability * 8) +   # Sustained negative momentum
                0.2 * np.tanh(volume_extreme - 1.5) +          # Extreme volume
                0.2 * oversold_condition                        # Oversold condition
            )
            
            # Normalize to [0, 1] range
            capitulation_score = (capitulation_score + 1) / 2
            
            df['capitulation_indicator'] = capitulation_score
            
            self.feature_names.append('capitulation_indicator')
            self.feature_descriptions['capitulation_indicator'] = 'Capitulation selling indicator'
            
        except Exception as e:
            logger.error(f"❌ Capitulation indicator failed: {e}")
            df['capitulation_indicator'] = 0.5
        
        return df
    
    def _add_greed_fear_balance(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add greed/fear balance indicator."""
        try:
            # Calculate greed and fear components
            price_momentum = df['close'].pct_change(window)
            volatility = df['close'].rolling(window).std() / df['close']
            
            # Greed component: positive momentum with low volatility
            greed_component = np.maximum(0, price_momentum) * (1 - volatility)
            
            # Fear component: negative momentum with high volatility
            fear_component = np.maximum(0, -price_momentum) * volatility
            
            # Greed/fear balance: positive = greed, negative = fear
            greed_fear_balance = greed_component - fear_component
            
            # Normalize to [-1, 1] range
            greed_fear_balance = np.tanh(greed_fear_balance * 5)
            
            df['greed_fear_balance'] = greed_fear_balance
            
            self.feature_names.append('greed_fear_balance')
            self.feature_descriptions['greed_fear_balance'] = 'Greed vs Fear balance'
            
        except Exception as e:
            logger.error(f"❌ Greed/fear balance failed: {e}")
            df['greed_fear_balance'] = 0.0
        
        return df
    
    def _add_crowd_sentiment(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add crowd sentiment indicator."""
        try:
            # Calculate crowd sentiment based on multiple factors
            price_momentum = df['close'].pct_change(window)
            volume_trend = df['volume'].pct_change(window)
            price_volatility = df['close'].rolling(window).std() / df['close']
            
            # Crowd sentiment components:
            # 1. Price momentum (positive = bullish crowd)
            # 2. Volume trend (increasing volume = stronger sentiment)
            # 3. Volatility (low volatility = confident crowd)
            # 4. Price consistency (consistent direction = strong sentiment)
            
            price_consistency = np.abs(price_momentum.rolling(window).mean()) / (price_volatility + 1e-8)
            
            # Combine components
            crowd_sentiment = (
                0.4 * np.tanh(price_momentum * 10) +      # Price momentum
                0.2 * np.tanh(volume_trend * 5) +         # Volume trend
                0.2 * (1 - price_volatility * 10) +       # Low volatility (confidence)
                0.2 * np.tanh(price_consistency * 2)      # Price consistency
            )
            
            # Normalize to [-1, 1] range
            crowd_sentiment = np.tanh(crowd_sentiment)
            
            df['crowd_sentiment'] = crowd_sentiment
            
            self.feature_names.append('crowd_sentiment')
            self.feature_descriptions['crowd_sentiment'] = 'Crowd sentiment analysis'
            
        except Exception as e:
            logger.error(f"❌ Crowd sentiment failed: {e}")
            df['crowd_sentiment'] = 0.0
        
        return df
    
    def _add_herd_behavior(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add herd behavior indicator."""
        try:
            # Calculate herd behavior based on crowd movement patterns
            price_momentum = df['close'].pct_change(window)
            volume_momentum = df['volume'].pct_change(window)
            
            # Herd behavior indicators:
            # 1. Momentum clustering (many people moving in same direction)
            # 2. Volume clustering (high volume with momentum)
            # 3. Momentum acceleration (increasing momentum = herd following)
            # 4. Momentum consistency (sustained momentum = herd behavior)
            
            momentum_acceleration = price_momentum.diff()
            momentum_consistency = price_momentum.rolling(window).std()
            
            # Herd behavior score
            herd_score = (
                0.3 * np.tanh(np.abs(price_momentum) * 10) +           # Momentum strength
                0.2 * np.tanh(volume_momentum * 5) +                   # Volume momentum
                0.2 * np.tanh(np.abs(momentum_acceleration) * 20) +    # Momentum acceleration
                0.3 * (1 - momentum_consistency * 10)                  # Momentum consistency (low std = herd)
            )
            
            # Normalize to [0, 1] range
            herd_score = (herd_score + 1) / 2
            
            df['herd_behavior'] = herd_score
            
            self.feature_names.append('herd_behavior')
            self.feature_descriptions['herd_behavior'] = 'Herd behavior indicator'
            
        except Exception as e:
            logger.error(f"❌ Herd behavior failed: {e}")
            df['herd_behavior'] = 0.5
        
        return df
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get a summary of psychology features."""
        return {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_descriptions': self.feature_descriptions,
            'psychology_params': self.psychology_params
        }
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate psychology features for quality."""
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
    
    def analyze_market_psychology(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall market psychology."""
        try:
            psychology_analysis = {}
            
            # Analyze dominant emotion
            if 'fomo_indicator' in df.columns and 'panic_indicator' in df.columns:
                fomo_mean = df['fomo_indicator'].mean()
                panic_mean = df['panic_indicator'].mean()
                
                if fomo_mean > 0.7:
                    psychology_analysis['dominant_emotion'] = 'fomo'
                elif panic_mean > 0.7:
                    psychology_analysis['dominant_emotion'] = 'panic'
                elif fomo_mean > 0.6:
                    psychology_analysis['dominant_emotion'] = 'greed'
                elif panic_mean > 0.6:
                    psychology_analysis['dominant_emotion'] = 'fear'
                else:
                    psychology_analysis['dominant_emotion'] = 'neutral'
            
            # Analyze crowd behavior
            if 'herd_behavior' in df.columns:
                herd_mean = df['herd_behavior'].mean()
                
                if herd_mean > 0.7:
                    psychology_analysis['crowd_behavior'] = 'strong_herd'
                elif herd_mean > 0.5:
                    psychology_analysis['crowd_behavior'] = 'moderate_herd'
                else:
                    psychology_analysis['crowd_behavior'] = 'individualistic'
            
            # Analyze sentiment
            if 'crowd_sentiment' in df.columns:
                sentiment_mean = df['crowd_sentiment'].mean()
                
                if sentiment_mean > 0.3:
                    psychology_analysis['sentiment'] = 'bullish'
                elif sentiment_mean < -0.3:
                    psychology_analysis['sentiment'] = 'bearish'
                else:
                    psychology_analysis['sentiment'] = 'neutral'
            
            return psychology_analysis
            
        except Exception as e:
            logger.error(f"❌ Market psychology analysis failed: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'psychology_features_enabled': True,
        'validation_threshold': 0.8
    }
    
    # Initialize psychology features
    psychology_features = PsychologyFeatures(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'open': np.random.uniform(1000, 2000, 1000),
        'high': np.random.uniform(1000, 2000, 1000),
        'low': np.random.uniform(1000, 2000, 1000),
        'close': np.random.uniform(1000, 2000, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    # Add psychology features
    enhanced_data = psychology_features.add_psychology_features(sample_data)
    
    # Get feature summary
    summary = psychology_features.get_feature_summary()
    print(f"Added {summary['total_features']} psychology features")
    
    # Validate features
    validation = psychology_features.validate_features(enhanced_data)
    print("Psychology feature validation completed")
    
    # Analyze market psychology
    psychology = psychology_features.analyze_market_psychology(enhanced_data)
    print(f"Market psychology: {psychology}") 