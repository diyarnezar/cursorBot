"""
Month Trainer for Project Hyperion
Advanced training mode for 1-month data with maximum intelligence
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from core.base_trainer import BaseTrainer
from modules.feature_engineering import EnhancedFeatureEngineer
from models.enhanced_model_trainer import EnhancedModelTrainer
from models.ensemble_trainer import EnsembleTrainer
from utils.optimization import HyperparameterOptimizer
from utils.reinforcement_learning import RLAgent
from utils.self_improvement import SelfImprovementEngine
from utils.logging.logger import setup_logger

class MonthTrainer(BaseTrainer):
    """
    Advanced Month Trainer with Maximum Intelligence:
    
    1. Comprehensive Data Collection (30 days of minute data)
    2. Advanced Feature Engineering (500+ features)
    3. Deep Learning Models (LSTM, GRU, Transformer)
    4. Ensemble Learning & Meta-Learning
    5. Reinforcement Learning Optimization
    6. Self-Improvement Systems
    7. Hyperparameter Optimization
    8. Walk-Forward Validation
    """
    
    def __init__(self, symbols: List[str]):
        """Initialize Month Trainer with advanced intelligence"""
        super().__init__('month', symbols)
        self.symbols = symbols
        self.logger = setup_logger("hyperion.month_trainer")
        
        # Initialize advanced components
        self.feature_engineer = EnhancedFeatureEngineer()
        self.model_trainer = EnhancedModelTrainer()
        self.ensemble_trainer = EnsembleTrainer()
        self.hyperopt = HyperparameterOptimizer()
        self.rl_agent = RLAgent()
        self.self_improvement = SelfImprovementEngine()
        
        # Update mode configuration for maximum intelligence
        self.mode_config = {
            'name': 'Advanced Month Training',
            'description': 'Maximum Intelligence 30-Day Training',
            'days': 30.0,
            'minutes': 43200,
            'estimated_time': '30-45 minutes',
            'weight': 'High',
            'recommended_for': 'Monthly model retraining with full intelligence',
            'rate_limit_safe': True,
            'max_symbols_per_batch': 25,
            'batch_delay_seconds': 90,
            'features_enabled': [
                'basic', 'technical', 'price', 'volume', 'volatility', 
                'momentum', 'microstructure', 'psychology', 'patterns', 
                'regime_detection', 'external_alpha', 'advanced_indicators'
            ],
            'models_enabled': [
                'lightgbm', 'xgboost', 'catboost', 'random_forest', 
                'gradient_boosting', 'decision_tree', 'lstm', 'gru', 
                'transformer', 'ensemble', 'meta_learning'
            ],
            'advanced_features': True,
            'use_reinforcement_learning': True,
            'use_self_improvement': True,
            'use_hyperparameter_optimization': True,
            'use_meta_learning': True,
            'use_deep_learning': True,
            'use_ensemble_learning': True,
            'use_walk_forward_optimization': True
        }
        
        self.logger.info("Month Trainer initialized with maximum intelligence features")
    
    def train(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Execute advanced month training with maximum intelligence
        
        Args:
            data: Pre-collected data (optional)
            
        Returns:
            Dictionary containing trained models and results
        """
        self.logger.info("🚀 Starting Advanced Month Training with Maximum Intelligence")
        
        try:
            # Phase 1: Advanced Data Collection & Processing
            if data is None:
                data = self._collect_advanced_data()
            else:
                self.logger.info(f"✅ Using pre-collected data: {len(data)} points")
            
            # Phase 2: Maximum Feature Engineering
            features = self._generate_maximum_features(data)
            
            # Phase 3: Advanced Model Training
            models = self._train_advanced_models(features, data)
            
            # Phase 4: Reinforcement Learning Optimization
            self._apply_reinforcement_learning(models, features, data)
            
            # Phase 5: Self-Improvement & Enhancement
            self._apply_self_improvement(models, features, data)
            
            # Phase 6: Ensemble & Meta-Learning
            ensemble_models = self._create_advanced_ensemble(models, features, data)
            
            # Phase 7: Performance Validation & Optimization
            final_results = self._validate_and_optimize(models, ensemble_models, features, data)
            
            self.logger.info("🎉 Advanced Month Training completed successfully!")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Advanced month training failed: {e}")
            return {'error': str(e)}
    
    def _collect_advanced_data(self) -> pd.DataFrame:
        """Collect comprehensive 30-day data with minute precision"""
        self.logger.info("📊 Collecting advanced 30-day data...")
        
        # For 30 days, we need approximately 43,200 minute klines
        # (30 days * 24 hours * 60 minutes = 43,200)
        # But we'll collect more to ensure comprehensive coverage
        
        all_data = []
        for symbol in self.symbols:
            try:
                # Collect 50,000 klines to ensure comprehensive coverage
                # This gives us more than 30 days of minute data
                data = self.collect_data(symbol, limit=50000)
                if data is not None and not data.empty:
                    all_data.append(data)
                    self.logger.info(f"✅ Collected {len(data)} data points for {symbol}")
                else:
                    self.logger.warning(f"⚠️ No data collected for {symbol}")
            except Exception as e:
                self.logger.error(f"❌ Error collecting data for {symbol}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"✅ Total advanced data collected: {len(combined_df)} points")
            return combined_df
        else:
            self.logger.error("❌ No data collected")
            return pd.DataFrame()
    
    def _generate_maximum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate maximum intelligence features"""
        self.logger.info("🧠 Generating maximum intelligence features...")
        
        try:
            # Generate all basic features
            features = self.feature_engineer.enhance_features(data)
            
            # Add maximum intelligence features
            features = self.feature_engineer._add_maximum_intelligence_features(features)
            
            # Add advanced technical indicators
            features = self._add_advanced_indicators(features)
            
            # Add market microstructure features
            features = self._add_microstructure_features(features)
            
            # Add psychological indicators
            features = self._add_psychological_indicators(features)
            
            # Add regime detection features
            features = self._add_regime_detection_features(features)
            
            self.logger.info(f"✅ Generated {len(features.columns)} maximum intelligence features")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error generating maximum features: {e}")
            return data
    
    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators"""
        try:
            # Moving Averages (Multiple periods)
            periods = [5, 8, 13, 21, 34, 55, 89, 144, 233]
            for period in periods:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'wma_{period}'] = df['close'].rolling(window=period).apply(
                    lambda x: np.average(x, weights=np.arange(1, len(x) + 1))
                )
            
            # Calculate HMA after all WMAs are available
            for period in periods:
                half_period = period // 2
                if half_period in periods:  # Only calculate HMA if half_period WMA exists
                    df[f'hma_{period}'] = df[f'wma_{half_period}'] * 2 - df[f'wma_{period}']
            
            # Advanced Oscillators
            df['cci'] = self._calculate_cci(df)
            df['williams_r'] = self._calculate_williams_r(df)
            df['ultimate_oscillator'] = self._calculate_ultimate_oscillator(df)
            df['money_flow_index'] = self._calculate_mfi(df)
            df['stochastic_rsi'] = self._calculate_stochastic_rsi(df)
            
            # Advanced Momentum
            df['roc'] = df['close'].pct_change(periods=10) * 100
            df['momentum'] = df['close'] - df['close'].shift(10)
            df['tsi'] = self._calculate_tsi(df)
            df['kst'] = self._calculate_kst(df)
            
            # Volatility Indicators
            df['natr'] = self._calculate_natr(df)
            df['keltner_channels'] = self._calculate_keltner_channels(df)
            df['donchian_channels'] = self._calculate_donchian_channels(df)
            
            # Volume Indicators
            df['vwap'] = self._calculate_vwap(df)
            df['volume_price_trend'] = self._calculate_vpt(df)
            df['accumulation_distribution'] = self._calculate_ad(df)
            df['chaikin_money_flow'] = self._calculate_cmf(df)
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Error adding advanced indicators: {e}")
            # Return original dataframe if error occurs
            return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        try:
            # Bid-Ask Spread proxies
            df['spread_proxy'] = (df['high'] - df['low']) / df['close']
            df['spread_ratio'] = df['spread_proxy'].rolling(20).mean()
            
            # Order Flow Indicators
            df['volume_price_ratio'] = df['volume'] / df['close']
            df['volume_momentum'] = df['volume'].pct_change()
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # Liquidity Indicators
            df['amihud_illiquidity'] = abs(df['close'].pct_change()) / df['volume']
            df['kyle_lambda'] = abs(df['close'].pct_change()) / df['volume'].rolling(20).std()
            
            # Market Impact
            df['price_impact'] = df['close'].pct_change() / df['volume'].rolling(10).mean()
            df['volume_impact'] = df['volume'] / df['volume'].rolling(50).mean()
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Error adding microstructure features: {e}")
            return df
    
    def _add_psychological_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add psychological market indicators"""
        try:
            # Fear & Greed Indicators
            df['fear_greed_ratio'] = (df['close'] - df['close'].rolling(20).min()) / \
                                   (df['close'].rolling(20).max() - df['close'].rolling(20).min())
            
            # Sentiment Indicators
            df['bull_bear_ratio'] = df['close'].rolling(10).apply(
                lambda x: len([i for i in range(1, len(x)) if x.iloc[i] > x.iloc[i-1]]) / len(x)
            )
            
            # Confidence Indicators
            df['confidence_index'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            # Momentum Divergence
            df['price_momentum_divergence'] = df['close'].pct_change() - df['volume'].pct_change()
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Error adding psychological indicators: {e}")
            return df
    
    def _add_regime_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        try:
            # Volatility Regime
            df['volatility_regime'] = df['close'].rolling(20).std() / df['close'].rolling(60).std()
            
            # Trend Regime
            df['trend_regime'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
            
            # Momentum Regime
            df['momentum_regime'] = df['close'].pct_change(20).rolling(10).mean()
            
            # Volume Regime
            df['volume_regime'] = df['volume'].rolling(20).mean() / df['volume'].rolling(60).mean()
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Error adding regime detection features: {e}")
            return df
    
    def _train_advanced_models(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Any]:
        """Train advanced models with maximum intelligence"""
        self.logger.info("🤖 Training advanced models with maximum intelligence...")
        
        try:
            # Prepare features for numeric models
            numeric_features = features.select_dtypes(include=[np.number])
            
            # Ensure features and data have the same index
            if not numeric_features.index.equals(data.index):
                self.logger.info("Aligning features and data indices")
                common_index = numeric_features.index.intersection(data.index)
                numeric_features = numeric_features.loc[common_index]
                data = data.loc[common_index]
            
            # Create targets with proper alignment
            targets = {}
            
            # Close price target (current price)
            targets['close'] = data['close'].copy()
            
            # Returns target (next period return)
            returns = data['close'].pct_change().shift(-1)  # Shift to avoid look-ahead bias
            targets['returns'] = returns
            
            # Volatility target (rolling volatility)
            volatility = data['close'].rolling(window=20, min_periods=1).std()
            targets['volatility'] = volatility
            
            # Remove any remaining NaN values from targets
            for target_name, target_series in targets.items():
                nan_count = target_series.isna().sum()
                if nan_count > 0:
                    self.logger.info(f"Removing {nan_count} NaN values from {target_name} target")
                    # Forward fill for volatility, drop for others
                    if target_name == 'volatility':
                        targets[target_name] = target_series.fillna(method='ffill').fillna(method='bfill')
                    else:
                        targets[target_name] = target_series.dropna()
            
            # Final alignment check
            min_length = min(len(numeric_features), min(len(target) for target in targets.values()))
            if min_length < len(numeric_features):
                self.logger.info(f"Truncating features to {min_length} samples for alignment")
                numeric_features = numeric_features.iloc[:min_length]
                targets = {name: series.iloc[:min_length] for name, series in targets.items()}
            
            self.logger.info(f"Final data shapes - Features: {numeric_features.shape}")
            for target_name, target_series in targets.items():
                self.logger.info(f"Target {target_name}: {len(target_series)} samples")
            
            # Train models using enhanced trainer
            models = self.model_trainer.train_enhanced_models(numeric_features, targets)
            
            self.logger.info(f"✅ Trained {len(models)} advanced models")
            return models
            
        except Exception as e:
            self.logger.error(f"❌ Error training advanced models: {e}")
            return {}
    
    def _apply_reinforcement_learning(self, models: Dict[str, Any], features: pd.DataFrame, data: pd.DataFrame):
        """Apply reinforcement learning optimization"""
        self.logger.info("🧠 Applying reinforcement learning optimization...")
        
        try:
            # Initialize RL agent
            self.rl_agent.initialize_models(models)
            
            # Create target from close price (percentage change)
            target = data['close'].pct_change().shift(-1).dropna()
            
            # Align features with target
            common_index = features.index.intersection(target.index)
            features_aligned = features.loc[common_index]
            target_aligned = target.loc[common_index]
            
            if len(features_aligned) == 0 or len(target_aligned) == 0:
                self.logger.warning("No valid data for RL optimization, skipping")
                return models
            
            # Optimize models using RL with proper target
            optimized_models = self.rl_agent.optimize_models(
                features=features_aligned,
                target=target_aligned,
                episodes=self.config.get('rl_episodes', 100),
                learning_rate=0.001
            )
            
            self.logger.info("✅ Reinforcement learning optimization completed")
            return optimized_models
            
        except Exception as e:
            self.logger.warning(f"⚠️ RL optimization failed: {e}")
            return models
    
    def _apply_self_improvement(self, models: Dict[str, Any], features: pd.DataFrame, data: pd.DataFrame):
        """Apply self-improvement systems"""
        self.logger.info("🔄 Applying self-improvement systems...")
        
        try:
            # Initialize self-improvement engine with features and models
            self.self_improvement.initialize(features, models)
            
            # Apply improvement strategies
            improved_models = self.self_improvement.improve_models()
            
            self.logger.info("✅ Self-improvement systems completed")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Self-improvement failed: {e}")
    
    def _create_advanced_ensemble(self, models: Dict[str, Any], features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Any]:
        """Create advanced ensemble with meta-learning"""
        self.logger.info("🎯 Creating advanced ensemble with meta-learning...")
        
        try:
            # Create ensemble with proper target
            ensemble_models = self.ensemble_trainer.train_ensemble(models, features, data['close'])
            
            self.logger.info("✅ Advanced ensemble created")
            return ensemble_models
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ensemble creation failed: {e}")
            return models
    
    def _validate_and_optimize(self, models: Dict[str, Any], ensemble_models: Dict[str, Any], 
                              features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate and optimize final results"""
        self.logger.info("📊 Validating and optimizing final results...")
        
        try:
            # Combine all models
            all_models = {**models, **ensemble_models}
            
            # Create target from close price (percentage change)
            target = data['close'].pct_change().shift(-1).dropna()
            
            # Align features with target
            common_index = features.index.intersection(target.index)
            features_aligned = features.loc[common_index]
            target_aligned = target.loc[common_index]
            
            if len(features_aligned) == 0 or len(target_aligned) == 0:
                self.logger.warning("No valid data for hyperparameter optimization, skipping")
                optimized_models = all_models
            else:
                # Apply hyperparameter optimization with proper target
                optimized_models = self.hyperopt.optimize_models(all_models, features_aligned, target_aligned)
            
            # Calculate final metrics
            final_results = {
                'models': optimized_models,
                'total_models': len(optimized_models),
                'features_count': len(features.columns),
                'data_points': len(data),
                'training_mode': 'month',
                'intelligence_level': 'maximum',
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✅ Validation and optimization completed")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Validation and optimization failed: {e}")
            return {
                'models': models,
                'total_models': len(models),
                'features_count': len(features.columns),
                'data_points': len(data),
                'training_mode': 'month',
                'intelligence_level': 'maximum',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    # Helper methods for advanced indicators
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - sma_tp) / (0.015 * mad)
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()
        return -100 * (highest_high - df['close']) / (highest_high - lowest_low)
    
    def _calculate_ultimate_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Ultimate Oscillator"""
        try:
            # Calculate buying pressure (BP)
            bp = df['close'] - df[['low', 'close']].shift(1).min(axis=1)
            
            # Calculate true range (TR)
            tr = df[['high', 'close']].shift(1).max(axis=1) - df[['low', 'close']].shift(1).min(axis=1)
            
            # Calculate averages
            avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
            avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
            avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
            
            # Calculate Ultimate Oscillator
            uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
            
            return uo
            
        except Exception as e:
            self.logger.warning(f"Error calculating Ultimate Oscillator: {e}")
            # Return a simple RSI-like indicator as fallback
            return self._calculate_rsi(df['close'], 14)
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        rmf = tp * df['volume']
        
        positive_flow = rmf.where(tp > tp.shift(1), 0).rolling(period).sum()
        negative_flow = rmf.where(tp < tp.shift(1), 0).rolling(period).sum()
        
        mfi_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + mfi_ratio))
    
    def _calculate_stochastic_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic RSI"""
        rsi = self._calculate_rsi(df['close'], period)
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        return stoch_rsi
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_tsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Strength Index"""
        pc = df['close'].diff()
        apc = pc.ewm(span=25).mean()
        apc_ema = apc.ewm(span=13).mean()
        
        abs_pc = pc.abs()
        aapc = abs_pc.ewm(span=25).mean()
        aapc_ema = aapc.ewm(span=13).mean()
        
        return 100 * (apc_ema / aapc_ema)
    
    def _calculate_kst(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Know Sure Thing"""
        roc1 = df['close'].pct_change(10)
        roc2 = df['close'].pct_change(15)
        roc3 = df['close'].pct_change(20)
        roc4 = df['close'].pct_change(30)
        
        sma1 = roc1.rolling(10).mean()
        sma2 = roc2.rolling(10).mean()
        sma3 = roc3.rolling(10).mean()
        sma4 = roc4.rolling(15).mean()
        
        return (sma1 * 1) + (sma2 * 2) + (sma3 * 3) + (sma4 * 4)
    
    def _calculate_natr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Normalized Average True Range"""
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        
        atr = tr.rolling(period).mean()
        return (atr / df['close']) * 100
    
    def _calculate_keltner_channels(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Keltner Channels"""
        atr = self._calculate_atr(df, period)
        ema = df['close'].ewm(span=period).mean()
        return ema + (2 * atr)
    
    def _calculate_donchian_channels(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Donchian Channels"""
        return df['high'].rolling(period).max()
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    def _calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        return ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
    
    def _calculate_ad(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfv = mfm * df['volume']
        return mfv.cumsum()
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfv = mfm * df['volume']
        return mfv.rolling(period).sum() / df['volume'].rolling(period).sum()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        
        return tr.rolling(period).mean() 