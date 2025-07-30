"""
🏗️ Deep Stacking Ensemble Module

This module implements deep stacking ensemble methods for advanced
model combination and prediction in cryptocurrency trading.

Author: Hyperion Trading System
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure logging
logger = logging.getLogger(__name__)

class DeepStackingEnsemble:
    """
    🏗️ Deep Stacking Ensemble Generator
    
    Implements deep stacking ensemble methods for advanced model
    combination and prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the deep stacking ensemble.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.base_models = {}
        self.meta_models = {}
        self.scalers = {}
        self.ensemble_weights = {}
        self.performance_metrics = {}
        
        # Ensemble configurations
        self.ensemble_configs = {
            'base_models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'gradient_boosting': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'lightgbm': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'catboost': {
                    'iterations': 100,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'verbose': False
                },
                'svr': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'epsilon': 0.1
                }
            },
            'meta_models': {
                'neural_network': {
                    'layers': [64, 32, 16],
                    'dropout': 0.2,
                    'learning_rate': 0.001
                },
                'ridge_regression': {
                    'alpha': 1.0
                },
                'lasso_regression': {
                    'alpha': 0.1
                }
            }
        }
        
        logger.info("🏗️ Deep Stacking Ensemble initialized")
    
    def create_base_models(self) -> Dict[str, Any]:
        """Create all base models for the ensemble."""
        try:
            models = {}
            config = self.ensemble_configs['base_models']
            
            # Random Forest
            models['random_forest'] = RandomForestRegressor(**config['random_forest'])
            
            # Gradient Boosting
            models['gradient_boosting'] = GradientBoostingRegressor(**config['gradient_boosting'])
            
            # LightGBM
            models['lightgbm'] = lgb.LGBMRegressor(**config['lightgbm'])
            
            # XGBoost
            models['xgboost'] = xgb.XGBRegressor(**config['xgboost'])
            
            # CatBoost
            models['catboost'] = CatBoostRegressor(**config['catboost'])
            
            # SVR
            models['svr'] = SVR(**config['svr'])
            
            # Store base models
            self.base_models = models
            
            logger.info(f"✅ Created {len(models)} base models")
            return models
            
        except Exception as e:
            logger.error(f"❌ Failed to create base models: {e}")
            return {}
    
    def create_meta_models(self) -> Dict[str, Any]:
        """Create meta-models for stacking."""
        try:
            models = {}
            config = self.ensemble_configs['meta_models']
            
            # Neural Network Meta-Model
            nn_config = config['neural_network']
            nn_model = Sequential()
            
            for i, units in enumerate(nn_config['layers']):
                if i == 0:
                    nn_model.add(Dense(units, activation='relu', input_shape=(len(self.base_models),)))
                else:
                    nn_model.add(Dense(units, activation='relu'))
                
                nn_model.add(BatchNormalization())
                nn_model.add(Dropout(nn_config['dropout']))
            
            nn_model.add(Dense(1))
            
            optimizer = Adam(learning_rate=nn_config['learning_rate'])
            nn_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            models['neural_network'] = nn_model
            
            # Ridge Regression
            models['ridge_regression'] = Ridge(**config['ridge_regression'])
            
            # Lasso Regression
            models['lasso_regression'] = Lasso(**config['lasso_regression'])
            
            # Store meta models
            self.meta_models = models
            
            logger.info(f"✅ Created {len(models)} meta models")
            return models
            
        except Exception as e:
            logger.error(f"❌ Failed to create meta models: {e}")
            return {}
    
    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train all base models."""
        try:
            base_predictions = {}
            
            for name, model in self.base_models.items():
                logger.info(f"🏋️ Training base model: {name}")
                
                if name == 'svr':
                    # Scale features for SVR
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_train)
                    model.fit(X_scaled, y_train)
                    pred = model.predict(X_scaled)
                else:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_train)
                
                base_predictions[name] = pred
                
                # Calculate individual model performance
                mse = mean_squared_error(y_train, pred)
                r2 = r2_score(y_train, pred)
                
                self.performance_metrics[name] = {
                    'mse': mse,
                    'r2_score': r2,
                    'rmse': np.sqrt(mse)
                }
                
                logger.info(f"✅ {name} trained - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
            
            return base_predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to train base models: {e}")
            return {}
    
    def generate_meta_features(self, X: np.ndarray, y: np.ndarray, 
                             cv_folds: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate meta-features using cross-validation."""
        try:
            from sklearn.model_selection import KFold
            
            n_samples = len(X)
            meta_features = np.zeros((n_samples, len(self.base_models)))
            
            # K-fold cross-validation for meta-feature generation
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                logger.info(f"🔄 Generating meta-features for fold {fold + 1}/{cv_folds}")
                
                # Train base models on this fold
                for i, (name, model) in enumerate(self.base_models.items()):
                    if name == 'svr':
                        # Scale features for SVR
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_fold)
                        X_val_scaled = scaler.transform(X_val_fold)
                        model.fit(X_train_scaled, y_train_fold)
                        pred = model.predict(X_val_scaled)
                    else:
                        model.fit(X_train_fold, y_train_fold)
                        pred = model.predict(X_val_fold)
                    
                    meta_features[val_idx, i] = pred
            
            # Use the same data for meta-target (y)
            meta_target = y
            
            return meta_features, meta_target
            
        except Exception as e:
            logger.error(f"❌ Failed to generate meta-features: {e}")
            return np.array([]), np.array([])
    
    def train_meta_models(self, meta_features: np.ndarray, meta_target: np.ndarray) -> Dict[str, Any]:
        """Train meta-models using meta-features."""
        try:
            meta_predictions = {}
            
            for name, model in self.meta_models.items():
                logger.info(f"🏋️ Training meta-model: {name}")
                
                if name == 'neural_network':
                    # Neural network training with callbacks
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
                    ]
                    
                    # Reshape for neural network
                    meta_features_nn = meta_features.reshape(-1, len(self.base_models))
                    
                    model.fit(
                        meta_features_nn, meta_target,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=32,
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    pred = model.predict(meta_features_nn).flatten()
                else:
                    # Traditional ML models
                    model.fit(meta_features, meta_target)
                    pred = model.predict(meta_features)
                
                meta_predictions[name] = pred
                
                # Calculate meta-model performance
                mse = mean_squared_error(meta_target, pred)
                r2 = r2_score(meta_target, pred)
                
                self.performance_metrics[f'meta_{name}'] = {
                    'mse': mse,
                    'r2_score': r2,
                    'rmse': np.sqrt(mse)
                }
                
                logger.info(f"✅ {name} meta-model trained - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
            
            return meta_predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to train meta-models: {e}")
            return {}
    
    def predict_ensemble(self, X: np.ndarray, meta_model_name: str = 'neural_network') -> np.ndarray:
        """Make ensemble predictions."""
        try:
            # Get base model predictions
            base_predictions = np.zeros((len(X), len(self.base_models)))
            
            for i, (name, model) in enumerate(self.base_models.items()):
                if name == 'svr':
                    # Scale features for SVR
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                base_predictions[:, i] = pred
            
            # Get meta-model
            meta_model = self.meta_models.get(meta_model_name)
            if meta_model is None:
                logger.error(f"❌ Meta-model {meta_model_name} not found")
                return np.array([])
            
            # Make meta-prediction
            if meta_model_name == 'neural_network':
                base_predictions_nn = base_predictions.reshape(-1, len(self.base_models))
                ensemble_pred = meta_model.predict(base_predictions_nn).flatten()
            else:
                ensemble_pred = meta_model.predict(base_predictions)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"❌ Failed to make ensemble predictions: {e}")
            return np.array([])
    
    def calculate_ensemble_weights(self) -> Dict[str, float]:
        """Calculate optimal ensemble weights based on performance."""
        try:
            weights = {}
            total_performance = 0
            
            # Calculate weights based on R² scores
            for name, metrics in self.performance_metrics.items():
                if 'meta_' not in name:  # Only base models
                    r2_score = metrics['r2_score']
                    if r2_score > 0:  # Only positive R² scores
                        weights[name] = r2_score
                        total_performance += r2_score
            
            # Normalize weights
            if total_performance > 0:
                for name in weights:
                    weights[name] /= total_performance
            
            self.ensemble_weights = weights
            
            logger.info(f"✅ Calculated ensemble weights: {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate ensemble weights: {e}")
            return {}
    
    def weighted_ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted ensemble predictions."""
        try:
            if not self.ensemble_weights:
                self.calculate_ensemble_weights()
            
            # Get predictions from all base models
            predictions = {}
            for name, model in self.base_models.items():
                if name == 'svr':
                    # Scale features for SVR
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                predictions[name] = pred
            
            # Calculate weighted average
            weighted_pred = np.zeros(len(X))
            for name, weight in self.ensemble_weights.items():
                weighted_pred += weight * predictions[name]
            
            return weighted_pred
            
        except Exception as e:
            logger.error(f"❌ Failed to make weighted ensemble predictions: {e}")
            return np.array([])
    
    def evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        try:
            results = {}
            
            # Evaluate individual base models
            for name, model in self.base_models.items():
                if name == 'svr':
                    scaler = StandardScaler()
                    X_test_scaled = scaler.fit_transform(X_test)
                    pred = model.predict(X_test_scaled)
                else:
                    pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                
                results[name] = {
                    'mse': mse,
                    'r2_score': r2,
                    'rmse': np.sqrt(mse)
                }
            
            # Evaluate stacking ensemble
            for meta_name in self.meta_models.keys():
                pred = self.predict_ensemble(X_test, meta_name)
                if len(pred) > 0:
                    mse = mean_squared_error(y_test, pred)
                    r2 = r2_score(y_test, pred)
                    
                    results[f'stacking_{meta_name}'] = {
                        'mse': mse,
                        'r2_score': r2,
                        'rmse': np.sqrt(mse)
                    }
            
            # Evaluate weighted ensemble
            weighted_pred = self.weighted_ensemble_predict(X_test)
            if len(weighted_pred) > 0:
                mse = mean_squared_error(y_test, weighted_pred)
                r2 = r2_score(y_test, weighted_pred)
                
                results['weighted_ensemble'] = {
                    'mse': mse,
                    'r2_score': r2,
                    'rmse': np.sqrt(mse)
                }
            
            logger.info("✅ Ensemble evaluation completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate ensemble: {e}")
            return {}
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get a summary of the ensemble."""
        return {
            'base_models': list(self.base_models.keys()),
            'meta_models': list(self.meta_models.keys()),
            'ensemble_weights': self.ensemble_weights,
            'performance_metrics': self.performance_metrics,
            'configurations': self.ensemble_configs
        }
    
    def save_ensemble(self, filepath: str):
        """Save the ensemble to file."""
        try:
            import pickle
            
            ensemble_data = {
                'base_models': self.base_models,
                'meta_models': self.meta_models,
                'ensemble_weights': self.ensemble_weights,
                'performance_metrics': self.performance_metrics,
                'configurations': self.ensemble_configs
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(ensemble_data, f)
            
            logger.info(f"💾 Ensemble saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save ensemble: {e}")
    
    def load_ensemble(self, filepath: str):
        """Load the ensemble from file."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            self.base_models = ensemble_data['base_models']
            self.meta_models = ensemble_data['meta_models']
            self.ensemble_weights = ensemble_data['ensemble_weights']
            self.performance_metrics = ensemble_data['performance_metrics']
            self.ensemble_configs = ensemble_data['configurations']
            
            logger.info(f"📂 Ensemble loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load ensemble: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'ensemble_enabled': True,
        'cv_folds': 5,
        'meta_model': 'neural_network'
    }
    
    # Initialize deep stacking ensemble
    ensemble = DeepStackingEnsemble(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'target': np.random.randn(1000)
    })
    
    # Prepare data
    X = sample_data[['feature1', 'feature2', 'feature3']].values
    y = sample_data['target'].values
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train ensemble
    ensemble.create_base_models()
    ensemble.create_meta_models()
    
    # Train base models
    base_predictions = ensemble.train_base_models(X_train, y_train)
    
    # Generate meta-features
    meta_features, meta_target = ensemble.generate_meta_features(X_train, y_train)
    
    if len(meta_features) > 0:
        # Train meta-models
        meta_predictions = ensemble.train_meta_models(meta_features, meta_target)
        
        # Evaluate ensemble
        results = ensemble.evaluate_ensemble(X_test, y_test)
        
        # Print results
        for model_name, metrics in results.items():
            print(f"{model_name}: R² = {metrics['r2_score']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # Get ensemble summary
    summary = ensemble.get_ensemble_summary()
    print(f"Ensemble created with {len(summary['base_models'])} base models and {len(summary['meta_models'])} meta models") 