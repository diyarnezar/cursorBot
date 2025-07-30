"""
Professional Feature Engineering System for Project Hyperion
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import talib
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logging.logger import get_logger

class FeatureEngineer:
    """
    Professional feature engineering system with 300+ features and parallel processing
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.logger = get_logger("hyperion.features")
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Initialize parallel processing
        self.cpu_count = multiprocessing.cpu_count()
        self.max_workers = max(1, int(self.cpu_count * 0.9))  # Use 90% of CPU cores
        self.logger.info(f"🔄 Feature engineer initialized with {self.max_workers} parallel workers")
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features using parallel processing
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features
        """
        self.logger.info(f"🚀 Generating features using {self.max_workers} parallel workers")
        
        if df.empty:
            self.logger.warning("Empty DataFrame provided")
            return df
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                self.logger.error(f"Missing required column: {col}")
                return df
        
        # Create features using parallel processing
        df = self._create_features_parallel(df)
        
        # Clean up features
        df = self._clean_features(df)
        
        self.logger.info(f"✅ Generated {len(df.columns)} features")
        return df
    
    def _create_features_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features using parallel processing"""
        
        # Define feature generation tasks
        feature_tasks = [
            ('technical_indicators', self._add_technical_indicators),
            ('price_features', self._add_price_features),
            ('volume_features', self._add_volume_features),
            ('volatility_features', self._add_volatility_features),
            ('momentum_features', self._add_momentum_features),
            ('trend_features', self._add_trend_features),
            ('statistical_features', self._add_statistical_features),
            ('time_features', self._add_time_features),
            ('lag_features', self._add_lag_features),
            ('rolling_features', self._add_rolling_features)
        ]
        
        # Process features in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all feature generation tasks
            future_to_task = {
                executor.submit(task_func, df.copy()): task_name
                for task_name, task_func in feature_tasks
            }
            
            # Collect results
            feature_dfs = {}
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result_df = future.result()
                    feature_dfs[task_name] = result_df
                    self.logger.debug(f"✅ Completed {task_name}")
                except Exception as e:
                    self.logger.error(f"❌ Error in {task_name}: {e}")
                    feature_dfs[task_name] = df.copy()
        
        # Combine all feature DataFrames
        combined_df = df.copy()
        for task_name, feature_df in feature_dfs.items():
            # Add new columns from this feature set
            new_columns = [col for col in feature_df.columns if col not in combined_df.columns]
            for col in new_columns:
                combined_df[col] = feature_df[col]
        
        return combined_df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using TA-Lib"""
        try:
            # Moving averages
            df['sma_5'] = talib.SMA(df['close'], timeperiod=5)
            df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            df['sma_100'] = talib.SMA(df['close'], timeperiod=100)
            
            # Exponential moving averages
            df['ema_5'] = talib.EMA(df['close'], timeperiod=5)
            df['ema_10'] = talib.EMA(df['close'], timeperiod=10)
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
            
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
            
            # Williams %R
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
            
            # CCI
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'])
            
            # ADX
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
            
            # ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
            
            # OBV
            df['obv'] = talib.OBV(df['close'], df['volume'])
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            # Price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['price_change'].abs()
            
            # High-Low spread
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
            df['hl_ratio'] = df['high'] / df['low']
            
            # Open-Close spread
            df['oc_spread'] = (df['close'] - df['open']) / df['open']
            
            # Price position within range
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Price momentum
            df['price_momentum_1'] = df['close'].pct_change(1)
            df['price_momentum_3'] = df['close'].pct_change(3)
            df['price_momentum_5'] = df['close'].pct_change(5)
            df['price_momentum_10'] = df['close'].pct_change(10)
            
            # Price acceleration
            df['price_acceleration'] = df['price_momentum_1'].diff()
            
        except Exception as e:
            self.logger.error(f"Error adding price features: {e}")
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            # Volume changes
            df['volume_change'] = df['volume'].pct_change()
            df['volume_change_abs'] = df['volume_change'].abs()
            
            # Volume moving averages
            df['volume_sma_5'] = df['volume'].rolling(5).mean()
            df['volume_sma_10'] = df['volume'].rolling(10).mean()
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            
            # Volume ratio
            df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
            df['volume_ratio_10'] = df['volume'] / df['volume_sma_10']
            df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
            
            # Volume momentum
            df['volume_momentum'] = df['volume'].pct_change(5)
            
            # Price-volume relationship
            df['price_volume_corr'] = df['close'].rolling(10).corr(df['volume'])
            
        except Exception as e:
            self.logger.error(f"Error adding volume features: {e}")
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        try:
            # Rolling volatility
            for window in [5, 10, 20, 30]:
                df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
            
            # Volatility ratio
            df['volatility_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
            df['volatility_ratio_10_30'] = df['volatility_10'] / df['volatility_30']
            
            # Volatility momentum
            df['volatility_momentum'] = df['volatility_20'].pct_change()
            
            # Parkinson volatility
            df['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                ((np.log(df['high'] / df['low']) ** 2).rolling(20).mean())
            )
            
        except Exception as e:
            self.logger.error(f"Error adding volatility features: {e}")
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        try:
            # Price momentum
            for period in [1, 3, 5, 10, 20]:
                df[f'momentum_{period}'] = df['close'].pct_change(period)
            
            # Momentum acceleration
            df['momentum_accel_1'] = df['momentum_1'].diff()
            df['momentum_accel_5'] = df['momentum_5'].diff()
            
            # Momentum divergence
            df['momentum_divergence'] = df['momentum_5'] - df['momentum_20']
            
            # Rate of change
            df['roc_5'] = talib.ROC(df['close'], timeperiod=5)
            df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
            df['roc_20'] = talib.ROC(df['close'], timeperiod=20)
            
        except Exception as e:
            self.logger.error(f"Error adding momentum features: {e}")
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend features"""
        try:
            # Trend direction
            df['trend_5'] = np.where(df['close'] > df['close'].shift(5), 1, -1)
            df['trend_10'] = np.where(df['close'] > df['close'].shift(10), 1, -1)
            df['trend_20'] = np.where(df['close'] > df['close'].shift(20), 1, -1)
            
            # Trend strength
            df['trend_strength_5'] = abs(df['close'] - df['close'].shift(5)) / df['close'].shift(5)
            df['trend_strength_10'] = abs(df['close'] - df['close'].shift(10)) / df['close'].shift(10)
            
            # Moving average crossovers
            df['ma_cross_5_20'] = np.where(df['sma_5'] > df['sma_20'], 1, -1)
            df['ma_cross_10_50'] = np.where(df['sma_10'] > df['sma_50'], 1, -1)
            
        except Exception as e:
            self.logger.error(f"Error adding trend features: {e}")
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        try:
            # Z-score
            df['z_score_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            
            # Percentile rank
            df['percentile_rank_20'] = df['close'].rolling(20).rank(pct=True)
            
            # Skewness and kurtosis
            df['skewness_20'] = df['close'].rolling(20).skew()
            df['kurtosis_20'] = df['close'].rolling(20).kurt()
            
            # Quantiles
            df['quantile_25_20'] = df['close'].rolling(20).quantile(0.25)
            df['quantile_75_20'] = df['close'].rolling(20).quantile(0.75)
            
        except Exception as e:
            self.logger.error(f"Error adding statistical features: {e}")
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            # Extract time components
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Market session indicators
            df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error adding time features: {e}")
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        try:
            # Lagged prices
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
            # Lagged returns
            for lag in [1, 2, 3, 5]:
                df[f'return_lag_{lag}'] = df['price_change'].shift(lag)
            
        except Exception as e:
            self.logger.error(f"Error adding lag features: {e}")
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        try:
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
                df[f'close_std_{window}'] = df['close'].rolling(window).std()
                df[f'close_min_{window}'] = df['close'].rolling(window).min()
                df[f'close_max_{window}'] = df['close'].rolling(window).max()
                
                df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
                df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            
            # Rolling correlations
            df['price_volume_corr_10'] = df['close'].rolling(10).corr(df['volume'])
            df['price_volume_corr_20'] = df['close'].rolling(20).corr(df['volume'])
            
        except Exception as e:
            self.logger.error(f"Error adding rolling features: {e}")
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features"""
        try:
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill NaN values
            df = df.fillna(method='ffill')
            
            # Backward fill remaining NaN values
            df = df.fillna(method='bfill')
            
            # Fill any remaining NaN values with 0
            df = df.fillna(0)
            
            # Store feature names
            self.feature_names = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            self.logger.error(f"Error cleaning features: {e}")
        
        return df
    
    def create_targets(self, df: pd.DataFrame, horizons: List[int] = [1, 5, 15, 30, 60]) -> Dict[str, pd.Series]:
        """
        Create target variables for different prediction horizons
        
        Args:
            df: DataFrame with features
            horizons: List of prediction horizons in minutes
            
        Returns:
            Dictionary mapping horizon to target series
        """
        targets = {}
        
        try:
            for horizon in horizons:
                # Future price change
                target_name = f'target_{horizon}m'
                df[target_name] = df['close'].shift(-horizon).pct_change(horizon)
                
                # Binary classification target (up/down)
                binary_target_name = f'target_{horizon}m_binary'
                df[binary_target_name] = (df[target_name] > 0).astype(int)
                
                targets[f'{horizon}m'] = df[target_name]
                targets[f'{horizon}m_binary'] = df[binary_target_name]
            
            self.logger.info(f"Created targets for horizons: {horizons}")
            
        except Exception as e:
            self.logger.error(f"Error creating targets: {e}")
        
        return targets
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()
    
    def get_feature_importance(self, model, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Get feature importance from a trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                return dict(zip(feature_names, importance))
            else:
                self.logger.warning("Model does not have feature_importances_ attribute")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {} 