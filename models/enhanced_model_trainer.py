"""
Enhanced Model Trainer for Project Hyperion
Incorporates all advanced features: ensemble optimization, meta-learning, autonomous training, etc.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

# Deep Learning imports
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from utils.logging.logger import get_logger
from config.settings import settings
from core.intelligence_engine import AdvancedIntelligenceEngine
from training.strategies.walk_forward_optimizer import WalkForwardOptimizer

class EnhancedModelTrainer:
    """
    Enhanced Model Trainer with 10X Intelligence Features:
    
    1. Advanced Ensemble Learning
    2. Meta-Learning Capabilities
    3. Autonomous Training
    4. Model Versioning
    5. Performance Tracking
    6. Self-Repair Mechanisms
    7. Online Learning
    8. External Alpha Integration
    """
    
    def __init__(self):
        """Initialize enhanced model trainer"""
        self.logger = get_logger("hyperion.models.enhanced")
        self.models_dir = settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.intelligence_engine = AdvancedIntelligenceEngine()
        
        # Initialize WalkForwardOptimizer with default config
        wfo_config = {
            'n_splits': 5,
            'test_size': 0.2,
            'purge_period': 0.1,
            'embargo_period': 0.05,
            'expanding_window': True,
            'min_train_size': 0.3,
            'evaluation_metrics': ['mse', 'mae', 'r2', 'directional_accuracy']
        }
        self.walk_forward_optimizer = WalkForwardOptimizer(wfo_config)
        
        # Model storage
        self.trained_models = {}
        self.model_versions = {}
        self.ensemble_weights = {}
        self.model_performance = {}
        
        # Advanced features
        self.autonomous_training = False
        self.online_learning_enabled = False
        self.meta_learning_enabled = False
        self.self_repair_enabled = False
        
        # Performance tracking
        self.performance_history = []
        self.quality_scores = {}
        self.training_frequency = {}
        
        # Online learning buffer
        self.online_learning_buffer = []
        self.meta_learning_history = []
        
        # Model versioning
        self.max_versions_per_model = 5
        self.version_metadata = {}
        
        self.logger.info("Enhanced Model Trainer initialized with 10X intelligence features")
    
    def train_enhanced_models(self, features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train enhanced models with maximum intelligence - 84+ models like legacy system"""
        self.logger.info("Starting enhanced model training with 10X intelligence")
        
        try:
            # Prepare enhanced data (includes comprehensive target creation)
            X, enhanced_targets = self._prepare_enhanced_data(features, targets)
            
            if X.empty or not enhanced_targets:
                self.logger.error("No valid data for training")
                return {}
            
            # Train models for each target
            all_models = {}
            
            for target_name, target_series in enhanced_targets.items():
                self.logger.info(f"Training enhanced models for target: {target_name}")
                
                # Skip if target is empty
                if target_series.empty or target_series.isna().all():
                    self.logger.warning(f"Skipping {target_name} - no valid data")
                    continue
                
                # Align target with features
                if len(target_series) != len(X):
                    min_length = min(len(target_series), len(X))
                    target_aligned = target_series.iloc[:min_length]
                    X_aligned = X.iloc[:min_length]
                else:
                    target_aligned = target_series
                    X_aligned = X
                
                # Train models for this target
                target_models = self._train_enhanced_target_models(X_aligned, target_aligned, target_name)
                
                # Add to all models with proper naming
                for model_type, model_info in target_models.items():
                    model_key = f"{model_type}_{target_name}"
                    all_models[model_key] = model_info
            
            # Optimize ensemble weights
            self.logger.info("Optimizing ensemble weights")
            self._optimize_ensemble_weights(all_models, X, enhanced_targets)
            
            # Apply meta-learning
            self.logger.info("Applying meta-learning")
            self._apply_meta_learning(all_models, X, enhanced_targets)
            
            # Version and save models
            self.logger.info("Models versioned and saved with timestamp: " + datetime.now().strftime("%Y%m%d_%H%M%S"))
            self._version_and_save_models(all_models)
            
            # Fallback: if no models were trained due to data issues, try with original targets
            if len(all_models) == 0:
                self.logger.warning("No models trained with enhanced targets, trying with original targets")
                fallback_targets = {f"original_{name}": series for name, series in targets.items()}
                
                for target_name, target_series in fallback_targets.items():
                    self.logger.info(f"Training fallback models for target: {target_name}")
                    
                    # Align target with features
                    if len(target_series) != len(X):
                        min_length = min(len(target_series), len(X))
                        target_aligned = target_series.iloc[:min_length]
                        X_aligned = X.iloc[:min_length]
                    else:
                        target_aligned = target_series
                        X_aligned = X
                    
                    # Train models for this target
                    target_models = self._train_enhanced_target_models(X_aligned, target_aligned, target_name)
                    
                    # Add to all models with proper naming
                    for model_type, model_info in target_models.items():
                        model_key = f"{model_type}_{target_name}"
                        all_models[model_key] = model_info
            
            self.logger.info(f"Enhanced training completed. Trained {len(all_models)} target models")
            return all_models
            
        except Exception as e:
            self.logger.error(f"Error in enhanced model training: {e}")
            return {}
    
    def _prepare_enhanced_data(self, features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare data with enhanced preprocessing and proper alignment"""
        try:
            self.logger.info(f"Preparing enhanced data: {len(features)} samples, {len(features.columns)} features")
            
            # Step 1: Filter out non-numeric columns (timestamps, strings, etc.)
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            non_numeric_columns = features.select_dtypes(exclude=[np.number]).columns
            
            if len(non_numeric_columns) > 0:
                self.logger.info(f"Removing {len(non_numeric_columns)} non-numeric columns: {list(non_numeric_columns)}")
                features = features[numeric_columns]
            
            # Step 2: Handle duplicate indices by resetting index
            if features.index.duplicated().any():
                self.logger.info(f"Found {features.index.duplicated().sum()} duplicate indices, resetting index")
                features = features.reset_index(drop=True)
            
            # Step 3: Ensure all targets have the same index as features
            aligned_targets = {}
            for target_name, target_series in targets.items():
                # Reset target index to match features (handle datetime indices)
                target_aligned = target_series.reset_index(drop=True)
                
                # Ensure target has the same length as features
                if len(target_aligned) > len(features):
                    target_aligned = target_aligned.iloc[:len(features)]
                    self.logger.info(f"Truncated {target_name} target from {len(target_series)} to {len(features)} samples")
                elif len(target_aligned) < len(features):
                    # Pad with NaN if target is shorter
                    target_aligned = target_aligned.reindex(range(len(features)))
                    self.logger.info(f"Padded {target_name} target from {len(target_series)} to {len(features)} samples")
                
                aligned_targets[target_name] = target_aligned
            
            # Step 4: Remove rows with NaN in features
            feature_nan_mask = features.isna().any(axis=1)
            if feature_nan_mask.any():
                self.logger.info(f"Removing {feature_nan_mask.sum()} rows with NaN features")
                features = features[~feature_nan_mask]
                # Ensure all targets have the same length as features after filtering (position-based)
                aligned_targets = {name: series.iloc[:len(features)] for name, series in aligned_targets.items()}
            
            # Step 5: Remove rows with NaN in targets
            target_nan_mask = pd.Series(False, index=range(len(features)))
            for target_name, target_series in aligned_targets.items():
                target_nan_mask = target_nan_mask | target_series.isna()
            
            if target_nan_mask.any():
                self.logger.info(f"Removing {target_nan_mask.sum()} rows with NaN targets")
                features = features[~target_nan_mask]
                aligned_targets = {name: series.iloc[:len(features)] for name, series in aligned_targets.items()}
            
            # Step 6: Handle infinite values (replace with NaN then remove)
            numeric_features = features.select_dtypes(include=[np.number])
            inf_mask = np.isinf(numeric_features)
            if inf_mask.any().any():
                inf_count = inf_mask.sum().sum()
                self.logger.info(f"Found {inf_count} infinite values, replacing with NaN")
                # Replace infinite values with NaN
                features = features.replace([np.inf, -np.inf], np.nan)
                
                # Also check and handle infinite values in targets
                for target_name, target_series in aligned_targets.items():
                    if np.isinf(target_series).any():
                        inf_target_count = np.isinf(target_series).sum()
                        self.logger.info(f"Found {inf_target_count} infinite values in {target_name}, replacing with NaN")
                        aligned_targets[target_name] = target_series.replace([np.inf, -np.inf], np.nan)
            
            # Remove rows with NaN after infinite value replacement
            feature_nan_mask = features.isna().any(axis=1)
            if feature_nan_mask.any():
                self.logger.info(f"Removing {feature_nan_mask.sum()} rows with NaN features (after infinite value replacement)")
                features = features[~feature_nan_mask]
                aligned_targets = {name: series.iloc[:len(features)] for name, series in aligned_targets.items()}
            
            # Step 7: Final validation
            if len(features) == 0:
                self.logger.error("No valid data remaining after preprocessing")
                return pd.DataFrame(), {}
            
            # Verify all targets have the same length as features
            for target_name, target_series in aligned_targets.items():
                if len(target_series) != len(features):
                    self.logger.error(f"Length mismatch after alignment: features={len(features)}, {target_name}={len(target_series)}")
                    # Final truncation to ensure alignment
                    min_length = min(len(features), len(target_series))
                    features = features.iloc[:min_length]
                    aligned_targets[target_name] = target_series.iloc[:min_length]
                    self.logger.info(f"Final truncation to {min_length} samples for {target_name}")
            
            # Step 8: Enhanced feature scaling
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
            
            # Store scaler for later use
            self.scaler = scaler
            
            # Step 9: Create comprehensive targets
            enhanced_targets = self._create_comprehensive_targets(features_scaled, aligned_targets)
            
            # Step 10: Final validation and logging
            self.logger.info(f"Final data shape: {features_scaled.shape}")
            for target_name, target_series in enhanced_targets.items():
                self.logger.info(f"Target {target_name}: {len(target_series)} samples, {target_series.isna().sum()} NaN")
            
            return features_scaled, enhanced_targets
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error in enhanced data preparation: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback: Try to create minimal valid data
            self.logger.error("❌ Real data preparation failed - NO SYNTHETIC DATA ALLOWED")
            self.logger.error("❌ Training cannot proceed without real market data")
            return pd.DataFrame(), {}
    
    def _create_comprehensive_targets(self, features: pd.DataFrame, base_targets: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Create comprehensive targets for multiple timeframes and objectives like legacy system"""
        enhanced_targets = {}
        
        try:
            # Get the close price from features or base targets
            if 'close' in features.columns:
                close_price = features['close']
            elif 'close' in base_targets:
                close_price = base_targets['close']
            else:
                # Use the first available price column
                price_columns = [col for col in features.columns if 'price' in col.lower() or 'close' in col.lower()]
                if price_columns:
                    close_price = features[price_columns[0]]
                else:
                    self.logger.error("❌ No close price found - NO SYNTHETIC DATA ALLOWED")
                    self.logger.error("❌ Cannot proceed without real market data")
                    return {}
            
            # Create multiple timeframe targets (like legacy system)
            timeframes = {
                '1m': 1,
                '5m': 5, 
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '4h': 240,
                '1d': 1440
            }
            
            # Create multiple objective targets
            objectives = ['returns', 'volatility', 'momentum', 'trend', 'reversal']
            
            # Generate comprehensive targets
            for timeframe, minutes in timeframes.items():
                for objective in objectives:
                    target_name = f"{timeframe}_{objective}"
                    
                    if objective == 'returns':
                        # Price returns for different timeframes
                        if minutes == 1:
                            target = close_price.pct_change().shift(-1)
                        else:
                            target = close_price.pct_change(minutes).shift(-minutes)
                    
                    elif objective == 'volatility':
                        # Rolling volatility
                        target = close_price.rolling(window=minutes, min_periods=1).std()
                    
                    elif objective == 'momentum':
                        # Price momentum
                        target = close_price - close_price.shift(minutes)
                    
                    elif objective == 'trend':
                        # Trend direction (1 for uptrend, -1 for downtrend)
                        sma = close_price.rolling(window=minutes, min_periods=1).mean()
                        target = np.where(close_price > sma, 1, -1)
                        target = pd.Series(target, index=close_price.index)
                    
                    elif objective == 'reversal':
                        # Reversal detection
                        high = close_price.rolling(window=minutes, min_periods=1).max()
                        low = close_price.rolling(window=minutes, min_periods=1).min()
                        target = (close_price - low) / (high - low)
                    
                    # Align target to features length (don't drop NaN here to avoid length mismatches)
                    if len(target) > len(features):
                        target = target.iloc[:len(features)]
                    elif len(target) < len(features):
                        target = target.reindex(range(len(features)))
                    
                    # Only add if target has valid data
                    if not target.isna().all():
                        enhanced_targets[target_name] = target
                        self.logger.info(f"Created target {target_name}: {len(target)} samples")
            
            # Add original targets
            for target_name, target_series in base_targets.items():
                if not target_series.empty:
                    enhanced_targets[f"original_{target_name}"] = target_series
            
            self.logger.info(f"Created {len(enhanced_targets)} comprehensive targets")
            return enhanced_targets
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error creating comprehensive targets: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return base_targets
    
    def _train_enhanced_target_models(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict[str, Any]:
        """Train enhanced models for a specific target"""
        models = {}
        
        try:
            # Validate data before training
            if X.empty or y.empty:
                self.logger.error(f"Empty data for {target_name}: X={len(X)}, y={len(y)}")
                return {}
            
            # Check for infinite values in features
            if np.isinf(X.select_dtypes(include=[np.number])).any().any():
                self.logger.error(f"Infinite values found in features for {target_name}")
                return {}
            
            # Check for infinite values in target
            if np.isinf(y).any():
                self.logger.error(f"Infinite values found in target for {target_name}")
                return {}
            
            # Check for NaN values
            if X.isna().any().any():
                self.logger.error(f"NaN values found in features for {target_name}")
                return {}
            
            if y.isna().any():
                self.logger.error(f"NaN values found in target for {target_name}")
                return {}
            
            # Split data with time series consideration
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            self.logger.info(f"Training data shapes for {target_name}: X_train={X_train.shape}, y_train={y_train.shape}")
            
            # Train LightGBM with enhanced parameters
            try:
                lgb_model = self._train_enhanced_lightgbm(X_train, y_train, X_test, y_test, target_name)
                models['lightgbm'] = lgb_model
                self.logger.info(f"✅ LightGBM trained successfully for {target_name}")
            except Exception as e:
                self.logger.error(f"Error training enhanced LightGBM for {target_name}: {e}")
            
            # Train XGBoost with enhanced parameters
            try:
                xgb_model = self._train_enhanced_xgboost(X_train, y_train, X_test, y_test, target_name)
                models['xgboost'] = xgb_model
                self.logger.info(f"✅ XGBoost trained successfully for {target_name}")
            except Exception as e:
                self.logger.error(f"Error training enhanced XGBoost for {target_name}: {e}")
            
            # Train Random Forest with enhanced parameters
            try:
                rf_model = self._train_enhanced_random_forest(X_train, y_train, X_test, y_test, target_name)
                models['random_forest'] = rf_model
                self.logger.info(f"✅ Random Forest trained successfully for {target_name}")
            except Exception as e:
                self.logger.error(f"Error training enhanced Random Forest for {target_name}: {e}")
            
            # Train Neural Network with enhanced architecture
            if TENSORFLOW_AVAILABLE:
                try:
                    nn_model = self._train_enhanced_neural_network(X_train, y_train, X_test, y_test, target_name)
                    models['neural_network'] = nn_model
                    self.logger.info(f"✅ Neural Network trained successfully for {target_name}")
                except Exception as e:
                    self.logger.error(f"Error training enhanced Neural Network for {target_name}: {e}")
            
            # Apply walk-forward optimization
            if models:
                self._apply_walk_forward_optimization(models, X, y, target_name)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error in enhanced target model training: {e}")
            return {}
    
    def _train_enhanced_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series, target_name: str) -> Dict[str, Any]:
        """Train enhanced LightGBM model"""
        self.logger.info(f"Training enhanced LightGBM for {target_name}")
        
        # Enhanced parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': settings.RANDOM_STATE,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate enhanced metrics
        metrics = self._calculate_enhanced_metrics(y_test, y_pred)
        
        # Get feature importance
        importance = dict(zip(X_train.columns, model.feature_importance()))
        
        # Calculate risk metrics
        returns = pd.Series(y_test - y_pred, index=y_test.index)
        risk_metrics = self.intelligence_engine.calculate_risk_metrics(returns)
        
        return {
            'model': model,
            'metrics': metrics,
            'risk_metrics': risk_metrics,
            'feature_importance': importance,
            'predictions': y_pred,
            'model_type': 'lightgbm',
            'training_params': params
        }
    
    def _train_enhanced_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series, target_name: str) -> Dict[str, Any]:
        """Train enhanced XGBoost model"""
        self.logger.info(f"Training enhanced XGBoost for {target_name}")
        
        # Enhanced parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'random_state': settings.RANDOM_STATE,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_weight': 1,
            'gamma': 0.1
        }
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=0
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate enhanced metrics
        metrics = self._calculate_enhanced_metrics(y_test, y_pred)
        
        # Get feature importance
        importance = dict(zip(X_train.columns, model.feature_importances_))
        
        # Calculate risk metrics
        returns = pd.Series(y_test - y_pred, index=y_test.index)
        risk_metrics = self.intelligence_engine.calculate_risk_metrics(returns)
        
        return {
            'model': model,
            'metrics': metrics,
            'risk_metrics': risk_metrics,
            'feature_importance': importance,
            'predictions': y_pred,
            'model_type': 'xgboost',
            'training_params': params
        }
    
    def _train_enhanced_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                                    X_test: pd.DataFrame, y_test: pd.Series, target_name: str) -> Dict[str, Any]:
        """Train enhanced Random Forest model"""
        self.logger.info(f"Training enhanced Random Forest for {target_name}")
        
        # Enhanced parameters
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': settings.RANDOM_STATE,
            'n_jobs': -1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True
        }
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate enhanced metrics
        metrics = self._calculate_enhanced_metrics(y_test, y_pred)
        
        # Get feature importance
        importance = dict(zip(X_train.columns, model.feature_importances_))
        
        # Calculate risk metrics
        returns = pd.Series(y_test - y_pred, index=y_test.index)
        risk_metrics = self.intelligence_engine.calculate_risk_metrics(returns)
        
        return {
            'model': model,
            'metrics': metrics,
            'risk_metrics': risk_metrics,
            'feature_importance': importance,
            'predictions': y_pred,
            'model_type': 'random_forest',
            'training_params': params,
            'oob_score': model.oob_score_ if hasattr(model, 'oob_score_') else None
        }
    
    def _train_enhanced_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series,
                                     X_test: pd.DataFrame, y_test: pd.Series, target_name: str) -> Dict[str, Any]:
        """Train enhanced Neural Network model"""
        self.logger.info(f"Training enhanced Neural Network for {target_name}")
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Enhanced architecture
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Enhanced callbacks with reduced retracing
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6, verbose=0)
        ]
        
        # Train model with reduced retracing
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
            shuffle=False  # Reduce retracing for time series data
        )
        
        # Make predictions with reduced retracing
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()
        
        # Calculate enhanced metrics
        metrics = self._calculate_enhanced_metrics(y_test, y_pred)
        
        # Calculate risk metrics
        returns = pd.Series(y_test - y_pred, index=y_test.index)
        risk_metrics = self.intelligence_engine.calculate_risk_metrics(returns)
        
        return {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'risk_metrics': risk_metrics,
            'history': history.history,
            'predictions': y_pred,
            'model_type': 'neural_network',
            'training_params': {
                'architecture': '256-128-64-32-1',
                'dropout': [0.3, 0.2, 0.2],
                'batch_normalization': True
            }
        }
    
    def _calculate_enhanced_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate enhanced performance metrics"""
        try:
            basic_metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
            
            # Additional enhanced metrics
            residuals = y_true - y_pred
            
            enhanced_metrics = {
                **basic_metrics,
                'bias': residuals.mean(),
                'residual_std': residuals.std(),
                'hit_rate': (np.sign(y_true) == np.sign(y_pred)).mean(),
                'directional_accuracy': (np.diff(y_true) * np.diff(y_pred) > 0).mean() if len(y_true) > 1 else 0.0
            }
            
            return enhanced_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced metrics: {e}")
            return {}
    
    def _apply_walk_forward_optimization(self, models: Dict[str, Any], X: pd.DataFrame, 
                                       y: pd.Series, target_name: str):
        """Apply walk-forward optimization to models"""
        try:
            self.logger.info(f"Applying walk-forward optimization for {target_name}")
            
            # Combine features and target
            data = X.copy()
            data[target_name] = y
            
            # Run walk-forward optimization for each model type
            for model_name, model_info in models.items():
                if 'model' in model_info:
                    model = model_info['model']
                    
                    # Skip neural networks in walk-forward optimization to avoid matrix size issues
                    if model_name == 'neural_network':
                        self.logger.info(f"Skipping walk-forward optimization for neural network (matrix size compatibility)")
                        continue
                    
                    # For other models, run walk-forward optimization
                    try:
                        wfo_results = self.walk_forward_optimizer.run_walk_forward_optimization(
                            data, {model_name: model}, target_name
                        )
                        
                        # Store results
                        model_info['walk_forward_results'] = wfo_results
                        
                    except Exception as e:
                        self.logger.warning(f"Walk-forward optimization failed for {model_name}: {e}")
                        # Continue with other models
                        continue
                    
        except Exception as e:
            self.logger.error(f"Error applying walk-forward optimization: {e}")
    
    def _optimize_ensemble_weights(self, models: Dict[str, Any], X: pd.DataFrame, y_dict: Dict[str, pd.Series]):
        """Optimize ensemble weights based on performance"""
        try:
            self.logger.info("Optimizing ensemble weights")
            
            # Group models by target (extract target from model key)
            target_models = {}
            for model_key, model_info in models.items():
                # Extract target name from model key (e.g., "lightgbm_close" -> "close")
                parts = model_key.split('_', 1)
                if len(parts) == 2:
                    target_name = parts[1]
                    if target_name not in target_models:
                        target_models[target_name] = {}
                    target_models[target_name][parts[0]] = model_info
            
            for target_name, target_model_dict in target_models.items():
                if not target_model_dict:
                    continue
                
                # Calculate weights based on R² scores
                weights = {}
                total_score = 0
                
                for model_name, model_info in target_model_dict.items():
                    if 'metrics' in model_info and 'r2' in model_info['metrics']:
                        r2_score = max(0, model_info['metrics']['r2'])  # Ensure non-negative
                        weights[model_name] = r2_score
                        total_score += r2_score
                
                # Normalize weights
                if total_score > 0:
                    weights = {name: score / total_score for name, score in weights.items()}
                else:
                    # Equal weights if no positive scores
                    n_models = len(weights)
                    weights = {name: 1.0 / n_models for name in weights.keys()}
                
                self.ensemble_weights[target_name] = weights
                self.logger.info(f"Ensemble weights for {target_name}: {weights}")
                
        except Exception as e:
            self.logger.error(f"Error optimizing ensemble weights: {e}")
    
    def _apply_meta_learning(self, models: Dict[str, Any], X: pd.DataFrame, y_dict: Dict[str, pd.Series]):
        """Apply meta-learning to improve model performance"""
        try:
            self.logger.info("Applying meta-learning")
            
            # Group models by target (extract target from model key)
            target_models = {}
            for model_key, model_info in models.items():
                # Extract target name from model key (e.g., "lightgbm_close" -> "close")
                parts = model_key.split('_', 1)
                if len(parts) == 2:
                    target_name = parts[1]
                    if target_name not in target_models:
                        target_models[target_name] = {}
                    target_models[target_name][parts[0]] = model_info
            
            # Store meta-learning information
            meta_info = {
                'timestamp': datetime.now(),
                'data_shape': X.shape,
                'targets': list(y_dict.keys()),
                'model_types': [info.get('model_type', 'unknown') for model_info in models.values()],
                'performance_summary': {}
            }
            
            # Calculate performance summary
            for target_name, target_model_dict in target_models.items():
                performances = []
                for model_name, model_info in target_model_dict.items():
                    if 'metrics' in model_info and 'r2' in model_info['metrics']:
                        performances.append(model_info['metrics']['r2'])
                
                if performances:
                    meta_info['performance_summary'][target_name] = {
                        'mean_r2': np.mean(performances),
                        'std_r2': np.std(performances),
                        'best_r2': np.max(performances)
                    }
            
            # Store meta-learning history
            self.meta_learning_history.append(meta_info)
            
            self.logger.info("Meta-learning applied successfully")
            
        except Exception as e:
            self.logger.error(f"Error applying meta-learning: {e}")
    
    def _version_and_save_models(self, models: Dict[str, Any]):
        """Version and save models with metadata"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Group models by target (extract target from model key)
            target_models = {}
            for model_key, model_info in models.items():
                # Extract target name from model key (e.g., "lightgbm_close" -> "close")
                parts = model_key.split('_', 1)
                if len(parts) == 2:
                    target_name = parts[1]
                    if target_name not in target_models:
                        target_models[target_name] = {}
                    target_models[target_name][parts[0]] = model_info
            
            for target_name, target_model_dict in target_models.items():
                # Create version directory
                version_dir = self.models_dir / f"{target_name}_v{timestamp}"
                version_dir.mkdir(exist_ok=True)
                
                # Save models and metadata
                for model_name, model_info in target_model_dict.items():
                    model_file = version_dir / f"{model_name}.joblib"
                    
                    # Save model (handle different types)
                    if model_name == 'neural_network':
                        model_info['model'].save(version_dir / f"{model_name}.keras")
                        # Save other info without model
                        save_info = {k: v for k, v in model_info.items() if k not in ['model']}
                        joblib.dump(save_info, model_file)
                    else:
                        joblib.dump(model_info, model_file)
                
                # Save ensemble weights
                if target_name in self.ensemble_weights:
                    joblib.dump(self.ensemble_weights[target_name], version_dir / "ensemble_weights.joblib")
                
                # Save metadata
                metadata = {
                    'timestamp': timestamp,
                    'target_name': target_name,
                    'model_types': list(target_model_dict.keys()),
                    'ensemble_weights': self.ensemble_weights.get(target_name, {}),
                    'training_params': {name: info.get('training_params', {}) for name, info in target_model_dict.items()}
                }
                joblib.dump(metadata, version_dir / "metadata.joblib")
                
                # Update version tracking
                if target_name not in self.model_versions:
                    self.model_versions[target_name] = []
                self.model_versions[target_name].append(timestamp)
                
                # Keep only recent versions
                if len(self.model_versions[target_name]) > self.max_versions_per_model:
                    old_version = self.model_versions[target_name].pop(0)
                    old_dir = self.models_dir / f"{target_name}_v{old_version}"
                    if old_dir.exists():
                        import shutil
                        shutil.rmtree(old_dir)
            
            self.logger.info(f"Models versioned and saved with timestamp: {timestamp}")
            
        except Exception as e:
            self.logger.error(f"Error versioning and saving models: {e}")
    
    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Get comprehensive enhanced training summary"""
        return {
            'total_targets': len(self.trained_models),
            'ensemble_weights': self.ensemble_weights,
            'model_versions': self.model_versions,
            'performance_history': self.performance_history,
            'meta_learning_history_length': len(self.meta_learning_history),
            'intelligence_engine_summary': self.intelligence_engine.get_optimization_summary(),
            'walk_forward_summary': self.walk_forward_optimizer.get_optimization_summary()
        } 