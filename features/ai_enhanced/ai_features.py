"""
🤖 AI-Enhanced Features Module

This module implements 5 AI-enhanced features for maximum intelligence
in cryptocurrency trading. These features use advanced AI techniques
to provide superior prediction and analysis capabilities.

Author: Hyperion Trading System
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

# Configure logging
logger = logging.getLogger(__name__)

class AIEnhancedFeatures:
    """
    🤖 AI-Enhanced Features Generator
    
    Implements 5 AI-enhanced features for advanced prediction and analysis
    in cryptocurrency trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AI-enhanced features generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_names = []
        self.feature_descriptions = {}
        self.ai_models = {}
        self.scalers = {}
        
        # AI model configurations
        self.model_configs = {
            'trend_strength': {
                'model_type': 'lightgbm',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'volatility_forecast': {
                'model_type': 'xgboost',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'momentum': {
                'model_type': 'random_forest',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 8,
                    'random_state': 42
                }
            },
            'volume_signal': {
                'model_type': 'gradient_boosting',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 4,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'price_action': {
                'model_type': 'lightgbm',
                'params': {
                    'n_estimators': 150,
                    'max_depth': 7,
                    'learning_rate': 0.08,
                    'random_state': 42
                }
            }
        }
        
        logger.info("🤖 AI-Enhanced Features initialized")
    
    def add_ai_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all 5 AI-enhanced features to the dataframe.
        
        Args:
            df: Input dataframe with OHLCV data
            
        Returns:
            DataFrame with AI features added
        """
        try:
            logger.info("🤖 Adding AI-enhanced features...")
            
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
            
            # 1. AI Trend Strength
            df = self._add_ai_trend_strength(df, medium_window)
            
            # 2. AI Volatility Forecast
            df = self._add_ai_volatility_forecast(df, short_window)
            
            # 3. AI Momentum
            df = self._add_ai_momentum(df, medium_window)
            
            # 4. AI Volume Signal
            df = self._add_ai_volume_signal(df, short_window)
            
            # 5. AI Price Action
            df = self._add_ai_price_action(df, long_window)
            
            logger.info(f"✅ Added {len(self.feature_names)} AI-enhanced features")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to add AI features: {e}")
            return df
    
    def _add_ai_trend_strength(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add AI trend strength prediction feature."""
        try:
            # Prepare features for trend strength prediction
            features = self._prepare_trend_features(df, window)
            
            # Create target for trend strength
            target = self._create_trend_strength_target(df, window)
            
            # Train AI model for trend strength
            model, scaler = self._train_ai_model(features, target, 'trend_strength')
            
            # Make predictions
            predictions = self._make_ai_predictions(features, model, scaler, 'trend_strength')
            
            # Add feature to dataframe
            df['ai_trend_strength'] = predictions
            
            self.feature_names.append('ai_trend_strength')
            self.feature_descriptions['ai_trend_strength'] = 'AI-predicted trend strength'
            
        except Exception as e:
            logger.error(f"❌ AI trend strength failed: {e}")
            df['ai_trend_strength'] = 0.0
        
        return df
    
    def _add_ai_volatility_forecast(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add AI volatility forecasting feature."""
        try:
            # Prepare features for volatility forecasting
            features = self._prepare_volatility_features(df, window)
            
            # Create target for volatility forecast
            target = self._create_volatility_target(df, window)
            
            # Train AI model for volatility forecasting
            model, scaler = self._train_ai_model(features, target, 'volatility_forecast')
            
            # Make predictions
            predictions = self._make_ai_predictions(features, model, scaler, 'volatility_forecast')
            
            # Add feature to dataframe
            df['ai_volatility_forecast'] = predictions
            
            self.feature_names.append('ai_volatility_forecast')
            self.feature_descriptions['ai_volatility_forecast'] = 'AI volatility prediction'
            
        except Exception as e:
            logger.error(f"❌ AI volatility forecast failed: {e}")
            df['ai_volatility_forecast'] = 0.0
        
        return df
    
    def _add_ai_momentum(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add AI momentum analysis feature."""
        try:
            # Prepare features for momentum analysis
            features = self._prepare_momentum_features(df, window)
            
            # Create target for momentum
            target = self._create_momentum_target(df, window)
            
            # Train AI model for momentum analysis
            model, scaler = self._train_ai_model(features, target, 'momentum')
            
            # Make predictions
            predictions = self._make_ai_predictions(features, model, scaler, 'momentum')
            
            # Add feature to dataframe
            df['ai_momentum'] = predictions
            
            self.feature_names.append('ai_momentum')
            self.feature_descriptions['ai_momentum'] = 'AI momentum analysis'
            
        except Exception as e:
            logger.error(f"❌ AI momentum failed: {e}")
            df['ai_momentum'] = 0.0
        
        return df
    
    def _add_ai_volume_signal(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add AI volume signal analysis feature."""
        try:
            # Prepare features for volume signal analysis
            features = self._prepare_volume_features(df, window)
            
            # Create target for volume signal
            target = self._create_volume_signal_target(df, window)
            
            # Train AI model for volume signal analysis
            model, scaler = self._train_ai_model(features, target, 'volume_signal')
            
            # Make predictions
            predictions = self._make_ai_predictions(features, model, scaler, 'volume_signal')
            
            # Add feature to dataframe
            df['ai_volume_signal'] = predictions
            
            self.feature_names.append('ai_volume_signal')
            self.feature_descriptions['ai_volume_signal'] = 'AI volume analysis'
            
        except Exception as e:
            logger.error(f"❌ AI volume signal failed: {e}")
            df['ai_volume_signal'] = 0.0
        
        return df
    
    def _add_ai_price_action(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add AI price action prediction feature."""
        try:
            # Prepare features for price action prediction
            features = self._prepare_price_action_features(df, window)
            
            # Create target for price action
            target = self._create_price_action_target(df, window)
            
            # Train AI model for price action prediction
            model, scaler = self._train_ai_model(features, target, 'price_action')
            
            # Make predictions
            predictions = self._make_ai_predictions(features, model, scaler, 'price_action')
            
            # Add feature to dataframe
            df['ai_price_action'] = predictions
            
            self.feature_names.append('ai_price_action')
            self.feature_descriptions['ai_price_action'] = 'AI price action prediction'
            
        except Exception as e:
            logger.error(f"❌ AI price action failed: {e}")
            df['ai_price_action'] = 0.0
        
        return df
    
    def _prepare_trend_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Prepare features for trend strength prediction."""
        features = pd.DataFrame()
        
        # Price-based features
        features['price_change'] = df['close'].pct_change()
        features['price_ma_ratio'] = df['close'] / df['close'].rolling(window).mean()
        features['price_std'] = df['close'].rolling(window).std()
        features['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Volume-based features
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window).mean()
        features['volume_price_trend'] = df['volume'] * df['close'].pct_change()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(df['close'], window)
        features['macd'] = self._calculate_macd(df['close'])
        features['bollinger_position'] = (df['close'] - df['close'].rolling(window).mean()) / (df['close'].rolling(window).std() * 2)
        
        # Momentum features
        features['momentum'] = df['close'] - df['close'].shift(1)
        features['rate_of_change'] = df['close'].pct_change(window)
        
        # Clean features
        features = features.fillna(0.0)
        
        return features
    
    def _prepare_volatility_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Prepare features for volatility forecasting."""
        features = pd.DataFrame()
        
        # Historical volatility features
        returns = df['close'].pct_change()
        features['realized_volatility'] = returns.rolling(window).std()
        features['volatility_of_volatility'] = features['realized_volatility'].rolling(window).std()
        
        # Price-based volatility features
        features['high_low_volatility'] = (df['high'] - df['low']) / df['close']
        features['open_close_volatility'] = abs(df['close'] - df['open']) / df['open']
        
        # Volume-based volatility features
        features['volume_volatility'] = df['volume'].rolling(window).std() / df['volume'].rolling(window).mean()
        
        # Technical volatility features
        features['atr'] = self._calculate_atr(df, window)
        features['bollinger_width'] = (df['close'].rolling(window).mean() + 2 * df['close'].rolling(window).std()) - (df['close'].rolling(window).mean() - 2 * df['close'].rolling(window).std())
        
        # Clean features
        features = features.fillna(0.0)
        
        return features
    
    def _prepare_momentum_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Prepare features for momentum analysis."""
        features = pd.DataFrame()
        
        # Price momentum features
        features['price_momentum'] = df['close'].pct_change(window)
        features['price_acceleration'] = features['price_momentum'].diff()
        features['price_momentum_ma'] = features['price_momentum'].rolling(window).mean()
        
        # Volume momentum features
        features['volume_momentum'] = df['volume'].pct_change(window)
        features['volume_acceleration'] = features['volume_momentum'].diff()
        
        # Technical momentum features
        features['rsi_momentum'] = self._calculate_rsi(df['close'], window).diff()
        features['macd_momentum'] = self._calculate_macd(df['close']).diff()
        
        # Cross-momentum features
        features['price_volume_momentum'] = features['price_momentum'] * features['volume_momentum']
        
        # Clean features
        features = features.fillna(0.0)
        
        return features
    
    def _prepare_volume_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Prepare features for volume signal analysis."""
        features = pd.DataFrame()
        
        # Volume-based features
        features['volume_ma'] = df['volume'].rolling(window).mean()
        features['volume_std'] = df['volume'].rolling(window).std()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
        
        # Volume-price relationship
        features['volume_price_correlation'] = df['close'].rolling(window).corr(df['volume'].rolling(window))
        features['volume_weighted_price'] = (df['close'] * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
        
        # Volume patterns
        features['volume_trend'] = df['volume'].rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        features['volume_volatility'] = df['volume'].pct_change().rolling(window).std()
        
        # Clean features
        features = features.fillna(0.0)
        
        return features
    
    def _prepare_price_action_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Prepare features for price action prediction."""
        features = pd.DataFrame()
        
        # Candlestick patterns
        features['body_size'] = abs(df['close'] - df['open']) / df['open']
        features['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
        features['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
        
        # Price action patterns
        features['doji'] = np.where(features['body_size'] < 0.001, 1, 0)
        features['hammer'] = np.where((features['lower_shadow'] > 2 * features['body_size']) & (features['upper_shadow'] < features['body_size']), 1, 0)
        features['shooting_star'] = np.where((features['upper_shadow'] > 2 * features['body_size']) & (features['lower_shadow'] < features['body_size']), 1, 0)
        
        # Support and resistance
        features['support_level'] = df['low'].rolling(window).min()
        features['resistance_level'] = df['high'].rolling(window).max()
        features['support_distance'] = (df['close'] - features['support_level']) / df['close']
        features['resistance_distance'] = (features['resistance_level'] - df['close']) / df['close']
        
        # Clean features
        features = features.fillna(0.0)
        
        return features
    
    def _create_trend_strength_target(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Create target for trend strength prediction."""
        # Historical price movement (no future information)
        historical_return = df['close'].pct_change()
        
        # Trend strength based on historical consistency
        trend_consistency = historical_return.rolling(window).apply(lambda x: np.abs(np.corrcoef(x, range(len(x)))[0, 1]) if len(x) > 1 else 0)
        
        # Combine return and consistency
        trend_strength = historical_return * trend_consistency
        
        return trend_strength.fillna(0.0)
    
    def _create_volatility_target(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Create target for volatility forecasting."""
        # Historical realized volatility (no future information)
        historical_returns = df['close'].pct_change()
        historical_volatility = historical_returns.rolling(window).std()
        
        return historical_volatility.fillna(0.0)
    
    def _create_momentum_target(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Create target for momentum analysis."""
        # Historical momentum (no future information)
        historical_momentum = df['close'].pct_change()
        
        return historical_momentum.fillna(0.0)
    
    def _create_volume_signal_target(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Create target for volume signal analysis."""
        # Historical volume change (no future information)
        historical_volume_change = df['volume'].pct_change()
        
        return historical_volume_change.fillna(0.0)
    
    def _create_price_action_target(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Create target for price action prediction."""
        # Historical price action (direction and magnitude)
        historical_return = df['close'].pct_change()
        historical_direction = np.sign(historical_return)
        historical_magnitude = np.abs(historical_return)
        
        # Combined target
        price_action_target = historical_direction * historical_magnitude
        
        return price_action_target.fillna(0.0)
    
    def _train_ai_model(self, features: pd.DataFrame, target: pd.Series, model_name: str) -> Tuple[Any, StandardScaler]:
        """Train AI model for feature prediction."""
        try:
            # Remove rows with NaN values
            valid_mask = ~(features.isna().any(axis=1) | target.isna())
            X = features[valid_mask]
            y = target[valid_mask]
            
            if len(X) < 50:  # Need minimum data for training
                logger.warning(f"⚠️ Insufficient data for {model_name} training")
                return None, None
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Get model configuration
            config = self.model_configs[model_name]
            model_type = config['model_type']
            params = config['params']
            
            # Create and train model
            if model_type == 'lightgbm':
                model = lgb.LGBMRegressor(**params)
            elif model_type == 'xgboost':
                model = xgb.XGBRegressor(**params)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(**params)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(**params)
            else:
                model = LinearRegression()
            
            # Train model
            model.fit(X_scaled, y)
            
            # Store model and scaler
            self.ai_models[model_name] = model
            self.scalers[model_name] = scaler
            
            logger.info(f"✅ Trained {model_name} model")
            return model, scaler
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model_name} model: {e}")
            return None, None
    
    def _make_ai_predictions(self, features: pd.DataFrame, model: Any, scaler: StandardScaler, model_name: str) -> np.ndarray:
        """Make predictions using trained AI model."""
        try:
            if model is None or scaler is None:
                return np.zeros(len(features))
            
            # Remove rows with NaN values
            valid_mask = ~features.isna().any(axis=1)
            X = features[valid_mask]
            
            if len(X) == 0:
                return np.zeros(len(features))
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            # Create full prediction array
            full_predictions = np.zeros(len(features))
            full_predictions[valid_mask] = predictions
            
            return full_predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to make predictions for {model_name}: {e}")
            return np.zeros(len(features))
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window).mean()
        return atr
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get a summary of AI-enhanced features."""
        return {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_descriptions': self.feature_descriptions,
            'trained_models': list(self.ai_models.keys()),
            'model_configs': self.model_configs
        }
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate AI-enhanced features for quality."""
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


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'ai_features_enabled': True,
        'validation_threshold': 0.8
    }
    
    # Initialize AI features
    ai_features = AIEnhancedFeatures(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'open': np.random.uniform(1000, 2000, 1000),
        'high': np.random.uniform(1000, 2000, 1000),
        'low': np.random.uniform(1000, 2000, 1000),
        'close': np.random.uniform(1000, 2000, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    # Add AI features
    enhanced_data = ai_features.add_ai_features(sample_data)
    
    # Get feature summary
    summary = ai_features.get_feature_summary()
    print(f"Added {summary['total_features']} AI-enhanced features")
    
    # Validate features
    validation = ai_features.validate_features(enhanced_data)
    print("AI feature validation completed") 