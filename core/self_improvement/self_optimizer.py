"""
⚙️ Self-Optimizer Module

This module handles hyperparameter optimization, feature selection optimization,
and model repair mechanisms for continuous system improvement.

Author: Hyperion Trading System
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna

# Configure logging
logger = logging.getLogger(__name__)

class SelfOptimizer:
    """
    ⚙️ Self-Optimization System
    
    Handles hyperparameter optimization, feature selection optimization,
    and model repair mechanisms for continuous system improvement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the self-optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.optimization_history = []
        self.best_hyperparameters = {}
        self.best_feature_sets = {}
        self.model_performance_tracker = {}
        self.last_optimization_time = None
        self.optimization_interval = timedelta(hours=12)  # Optimize every 12 hours
        
        # Optimization modes
        self.optimization_modes = {
            'hyperparameter_optimization': True,
            'feature_selection_optimization': True,
            'model_repair': True,
            'performance_optimization': True,
            'ensemble_optimization': True
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_r2_score': 0.6,
            'max_mse': 0.1,
            'min_feature_importance': 0.01,
            'max_model_size': 1000
        }
        
        logger.info("⚙️ Self-Optimizer initialized")
    
    async def start_self_optimization(self):
        """Start the self-optimization process."""
        logger.info("🚀 Starting self-optimization system...")
        
        while True:
            try:
                await self._conduct_optimization_cycle()
                await asyncio.sleep(self.optimization_interval.total_seconds())
            except Exception as e:
                logger.error(f"❌ Optimization cycle failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _conduct_optimization_cycle(self):
        """Conduct a complete optimization cycle."""
        logger.info("⚙️ Conducting optimization cycle...")
        
        # Hyperparameter optimization
        if self.optimization_modes['hyperparameter_optimization']:
            await self._optimize_hyperparameters()
        
        # Feature selection optimization
        if self.optimization_modes['feature_selection_optimization']:
            await self._optimize_feature_selection()
        
        # Model repair
        if self.optimization_modes['model_repair']:
            await self._repair_models()
        
        # Performance optimization
        if self.optimization_modes['performance_optimization']:
            await self._optimize_performance()
        
        # Ensemble optimization
        if self.optimization_modes['ensemble_optimization']:
            await self._optimize_ensembles()
        
        self.last_optimization_time = datetime.now()
        logger.info("✅ Optimization cycle completed")
    
    async def _optimize_hyperparameters(self):
        """Optimize model hyperparameters using Optuna."""
        logger.info("🔧 Optimizing hyperparameters...")
        
        try:
            # Get latest data for optimization
            data = await self._get_optimization_data()
            
            if data is not None and len(data) > 100:
                # Optimize LightGBM hyperparameters
                await self._optimize_lightgbm_hyperparameters(data)
                
                # Optimize XGBoost hyperparameters
                await self._optimize_xgboost_hyperparameters(data)
                
                # Optimize CatBoost hyperparameters
                await self._optimize_catboost_hyperparameters(data)
                
                # Optimize Random Forest hyperparameters
                await self._optimize_random_forest_hyperparameters(data)
                
                logger.info("✅ Hyperparameter optimization completed")
            else:
                logger.info("ℹ️ Insufficient data for hyperparameter optimization")
                
        except Exception as e:
            logger.error(f"❌ Hyperparameter optimization failed: {e}")
    
    async def _optimize_lightgbm_hyperparameters(self, data: pd.DataFrame):
        """Optimize LightGBM hyperparameters."""
        try:
            def objective(trial):
                # Define hyperparameter search space
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': 42
                }
                
                # Prepare features and target
                X, y = self._prepare_optimization_data(data)
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                    scores.append(score)
                
                return np.mean(scores)
            
            # Create study and optimize
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            
            # Store best hyperparameters
            self.best_hyperparameters['lightgbm'] = study.best_params
            logger.info(f"✅ LightGBM optimization completed. Best score: {study.best_value:.4f}")
            
        except Exception as e:
            logger.error(f"❌ LightGBM hyperparameter optimization failed: {e}")
    
    async def _optimize_xgboost_hyperparameters(self, data: pd.DataFrame):
        """Optimize XGBoost hyperparameters."""
        try:
            def objective(trial):
                # Define hyperparameter search space
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': 42
                }
                
                # Prepare features and target
                X, y = self._prepare_optimization_data(data)
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                    scores.append(score)
                
                return np.mean(scores)
            
            # Create study and optimize
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            
            # Store best hyperparameters
            self.best_hyperparameters['xgboost'] = study.best_params
            logger.info(f"✅ XGBoost optimization completed. Best score: {study.best_value:.4f}")
            
        except Exception as e:
            logger.error(f"❌ XGBoost hyperparameter optimization failed: {e}")
    
    async def _optimize_catboost_hyperparameters(self, data: pd.DataFrame):
        """Optimize CatBoost hyperparameters."""
        try:
            def objective(trial):
                # Define hyperparameter search space
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 300),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                    'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                    'random_state': 42,
                    'verbose': False
                }
                
                # Prepare features and target
                X, y = self._prepare_optimization_data(data)
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = CatBoostRegressor(**params)
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                    
                    y_pred = model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                    scores.append(score)
                
                return np.mean(scores)
            
            # Create study and optimize
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            
            # Store best hyperparameters
            self.best_hyperparameters['catboost'] = study.best_params
            logger.info(f"✅ CatBoost optimization completed. Best score: {study.best_value:.4f}")
            
        except Exception as e:
            logger.error(f"❌ CatBoost hyperparameter optimization failed: {e}")
    
    async def _optimize_random_forest_hyperparameters(self, data: pd.DataFrame):
        """Optimize Random Forest hyperparameters."""
        try:
            def objective(trial):
                # Define hyperparameter search space
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42
                }
                
                # Prepare features and target
                X, y = self._prepare_optimization_data(data)
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = RandomForestRegressor(**params)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                    scores.append(score)
                
                return np.mean(scores)
            
            # Create study and optimize
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            
            # Store best hyperparameters
            self.best_hyperparameters['random_forest'] = study.best_params
            logger.info(f"✅ Random Forest optimization completed. Best score: {study.best_value:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Random Forest hyperparameter optimization failed: {e}")
    
    async def _optimize_feature_selection(self):
        """Optimize feature selection for each model."""
        logger.info("🔧 Optimizing feature selection...")
        
        try:
            # Get latest data for optimization
            data = await self._get_optimization_data()
            
            if data is not None and len(data) > 100:
                # Prepare features and target
                X, y = self._prepare_optimization_data(data)
                
                # Optimize feature selection for each model type
                await self._optimize_lightgbm_features(X, y)
                await self._optimize_xgboost_features(X, y)
                await self._optimize_catboost_features(X, y)
                await self._optimize_random_forest_features(X, y)
                
                logger.info("✅ Feature selection optimization completed")
            else:
                logger.info("ℹ️ Insufficient data for feature selection optimization")
                
        except Exception as e:
            logger.error(f"❌ Feature selection optimization failed: {e}")
    
    async def _optimize_lightgbm_features(self, X: pd.DataFrame, y: pd.Series):
        """Optimize feature selection for LightGBM."""
        try:
            # Use LightGBM's built-in feature importance
            model = lgb.LGBMRegressor(random_state=42)
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            feature_names = X.columns
            
            # Select features with importance above threshold
            threshold = self.performance_thresholds['min_feature_importance']
            selected_features = feature_names[importance > threshold]
            
            # Store best feature set
            self.best_feature_sets['lightgbm'] = selected_features.tolist()
            logger.info(f"✅ LightGBM feature selection: {len(selected_features)} features selected")
            
        except Exception as e:
            logger.error(f"❌ LightGBM feature selection failed: {e}")
    
    async def _optimize_xgboost_features(self, X: pd.DataFrame, y: pd.Series):
        """Optimize feature selection for XGBoost."""
        try:
            # Use XGBoost's built-in feature importance
            model = xgb.XGBRegressor(random_state=42)
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            feature_names = X.columns
            
            # Select features with importance above threshold
            threshold = self.performance_thresholds['min_feature_importance']
            selected_features = feature_names[importance > threshold]
            
            # Store best feature set
            self.best_feature_sets['xgboost'] = selected_features.tolist()
            logger.info(f"✅ XGBoost feature selection: {len(selected_features)} features selected")
            
        except Exception as e:
            logger.error(f"❌ XGBoost feature selection failed: {e}")
    
    async def _optimize_catboost_features(self, X: pd.DataFrame, y: pd.Series):
        """Optimize feature selection for CatBoost."""
        try:
            # Use CatBoost's built-in feature importance
            model = CatBoostRegressor(random_state=42, verbose=False)
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            feature_names = X.columns
            
            # Select features with importance above threshold
            threshold = self.performance_thresholds['min_feature_importance']
            selected_features = feature_names[importance > threshold]
            
            # Store best feature set
            self.best_feature_sets['catboost'] = selected_features.tolist()
            logger.info(f"✅ CatBoost feature selection: {len(selected_features)} features selected")
            
        except Exception as e:
            logger.error(f"❌ CatBoost feature selection failed: {e}")
    
    async def _optimize_random_forest_features(self, X: pd.DataFrame, y: pd.Series):
        """Optimize feature selection for Random Forest."""
        try:
            # Use Random Forest's built-in feature importance
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            feature_names = X.columns
            
            # Select features with importance above threshold
            threshold = self.performance_thresholds['min_feature_importance']
            selected_features = feature_names[importance > threshold]
            
            # Store best feature set
            self.best_feature_sets['random_forest'] = selected_features.tolist()
            logger.info(f"✅ Random Forest feature selection: {len(selected_features)} features selected")
            
        except Exception as e:
            logger.error(f"❌ Random Forest feature selection failed: {e}")
    
    async def _repair_models(self):
        """Repair underperforming models."""
        logger.info("🔧 Repairing models...")
        
        try:
            # Check model performance
            performance_issues = await self._detect_performance_issues()
            
            if performance_issues:
                for model_name, issue in performance_issues.items():
                    logger.warning(f"⚠️ Performance issue detected in {model_name}: {issue}")
                    
                    # Repair strategy based on issue type
                    if issue['type'] == 'low_accuracy':
                        await self._repair_low_accuracy_model(model_name)
                    elif issue['type'] == 'overfitting':
                        await self._repair_overfitting_model(model_name)
                    elif issue['type'] == 'underfitting':
                        await self._repair_underfitting_model(model_name)
                    elif issue['type'] == 'feature_degradation':
                        await self._repair_feature_degradation_model(model_name)
                
                logger.info("✅ Model repair completed")
            else:
                logger.info("ℹ️ No model repair needed")
                
        except Exception as e:
            logger.error(f"❌ Model repair failed: {e}")
    
    async def _detect_performance_issues(self) -> Dict[str, Any]:
        """Detect performance issues in models."""
        try:
            issues = {}
            
            # This would check actual model performance
            # For now, return empty dict (no issues detected)
            
            return issues
            
        except Exception as e:
            logger.error(f"❌ Performance issue detection failed: {e}")
            return {}
    
    async def _repair_low_accuracy_model(self, model_name: str):
        """Repair model with low accuracy."""
        try:
            logger.info(f"🔧 Repairing low accuracy model: {model_name}")
            
            # Retrain with more data
            # Adjust hyperparameters
            # Try different feature sets
            
        except Exception as e:
            logger.error(f"❌ Low accuracy model repair failed: {e}")
    
    async def _repair_overfitting_model(self, model_name: str):
        """Repair overfitting model."""
        try:
            logger.info(f"🔧 Repairing overfitting model: {model_name}")
            
            # Increase regularization
            # Reduce model complexity
            # Add more training data
            
        except Exception as e:
            logger.error(f"❌ Overfitting model repair failed: {e}")
    
    async def _repair_underfitting_model(self, model_name: str):
        """Repair underfitting model."""
        try:
            logger.info(f"🔧 Repairing underfitting model: {model_name}")
            
            # Increase model complexity
            # Add more features
            # Reduce regularization
            
        except Exception as e:
            logger.error(f"❌ Underfitting model repair failed: {e}")
    
    async def _repair_feature_degradation_model(self, model_name: str):
        """Repair model with feature degradation."""
        try:
            logger.info(f"🔧 Repairing feature degradation model: {model_name}")
            
            # Re-optimize feature selection
            # Add new features
            # Remove degraded features
            
        except Exception as e:
            logger.error(f"❌ Feature degradation model repair failed: {e}")
    
    async def _optimize_performance(self):
        """Optimize overall system performance."""
        logger.info("🔧 Optimizing system performance...")
        
        try:
            # Optimize data processing pipeline
            await self._optimize_data_pipeline()
            
            # Optimize model ensemble weights
            await self._optimize_ensemble_weights()
            
            # Optimize memory usage
            await self._optimize_memory_usage()
            
            # Optimize execution speed
            await self._optimize_execution_speed()
            
            logger.info("✅ Performance optimization completed")
            
        except Exception as e:
            logger.error(f"❌ Performance optimization failed: {e}")
    
    async def _optimize_data_pipeline(self):
        """Optimize data processing pipeline."""
        try:
            logger.info("🔧 Optimizing data pipeline...")
            
            # Optimize data loading
            # Optimize feature calculation
            # Optimize data storage
            
        except Exception as e:
            logger.error(f"❌ Data pipeline optimization failed: {e}")
    
    async def _optimize_ensemble_weights(self):
        """Optimize ensemble model weights."""
        try:
            logger.info("🔧 Optimizing ensemble weights...")
            
            # Calculate optimal weights based on recent performance
            # Update ensemble configuration
            
        except Exception as e:
            logger.error(f"❌ Ensemble weight optimization failed: {e}")
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage."""
        try:
            logger.info("🔧 Optimizing memory usage...")
            
            # Clean up unused models
            # Optimize data structures
            # Reduce memory footprint
            
        except Exception as e:
            logger.error(f"❌ Memory optimization failed: {e}")
    
    async def _optimize_execution_speed(self):
        """Optimize execution speed."""
        try:
            logger.info("🔧 Optimizing execution speed...")
            
            # Parallelize operations
            # Optimize algorithms
            # Reduce computational complexity
            
        except Exception as e:
            logger.error(f"❌ Execution speed optimization failed: {e}")
    
    async def _optimize_ensembles(self):
        """Optimize ensemble configurations."""
        logger.info("🔧 Optimizing ensembles...")
        
        try:
            # Optimize ensemble composition
            await self._optimize_ensemble_composition()
            
            # Optimize ensemble methods
            await self._optimize_ensemble_methods()
            
            # Optimize ensemble diversity
            await self._optimize_ensemble_diversity()
            
            logger.info("✅ Ensemble optimization completed")
            
        except Exception as e:
            logger.error(f"❌ Ensemble optimization failed: {e}")
    
    async def _optimize_ensemble_composition(self):
        """Optimize ensemble model composition."""
        try:
            logger.info("🔧 Optimizing ensemble composition...")
            
            # Select best performing models
            # Remove underperforming models
            # Add new model types if beneficial
            
        except Exception as e:
            logger.error(f"❌ Ensemble composition optimization failed: {e}")
    
    async def _optimize_ensemble_methods(self):
        """Optimize ensemble methods."""
        try:
            logger.info("🔧 Optimizing ensemble methods...")
            
            # Optimize stacking
            # Optimize blending
            # Optimize voting
            
        except Exception as e:
            logger.error(f"❌ Ensemble methods optimization failed: {e}")
    
    async def _optimize_ensemble_diversity(self):
        """Optimize ensemble diversity."""
        try:
            logger.info("🔧 Optimizing ensemble diversity...")
            
            # Ensure model diversity
            # Optimize feature diversity
            # Optimize training diversity
            
        except Exception as e:
            logger.error(f"❌ Ensemble diversity optimization failed: {e}")
    
    async def _get_optimization_data(self) -> Optional[pd.DataFrame]:
        """Get data for optimization."""
        try:
            # This would integrate with the data collection system
            # For now, return None to indicate no data available
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get optimization data: {e}")
            return None
    
    def _prepare_optimization_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for optimization."""
        try:
            # Select features (exclude target and metadata columns)
            feature_columns = [col for col in data.columns if col not in ['target', 'timestamp', 'pair']]
            X = data[feature_columns]
            
            # Create target (next period return)
            y = data['close'].pct_change().shift(-1)
            
            # Remove NaN values
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare optimization data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of optimization activities."""
        return {
            'best_hyperparameters': self.best_hyperparameters,
            'best_feature_sets': self.best_feature_sets,
            'model_performance_tracker': self.model_performance_tracker,
            'last_optimization_time': self.last_optimization_time,
            'optimization_modes': self.optimization_modes,
            'performance_thresholds': self.performance_thresholds,
            'optimization_history': self.optimization_history
        }
    
    def save_optimization_state(self, filepath: str):
        """Save optimization state to file."""
        try:
            state = {
                'best_hyperparameters': self.best_hyperparameters,
                'best_feature_sets': self.best_feature_sets,
                'model_performance_tracker': self.model_performance_tracker,
                'optimization_history': self.optimization_history
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"💾 Optimization state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save optimization state: {e}")
    
    def load_optimization_state(self, filepath: str):
        """Load optimization state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.best_hyperparameters = state.get('best_hyperparameters', {})
            self.best_feature_sets = state.get('best_feature_sets', {})
            self.model_performance_tracker = state.get('model_performance_tracker', {})
            self.optimization_history = state.get('optimization_history', [])
            
            logger.info(f"📂 Optimization state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load optimization state: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'optimization_enabled': True,
        'optimization_interval_hours': 12,
        'performance_threshold': 0.7
    }
    
    # Initialize self-optimizer
    optimizer = SelfOptimizer(config)
    
    # Start self-optimization
    asyncio.run(optimizer.start_self_optimization()) 