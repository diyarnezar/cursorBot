"""
🔄 Online Trainer Module

This module implements online training for incremental learning and
real-time model updates in cryptocurrency trading.

Author: Hyperion Trading System
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Configure logging
logger = logging.getLogger(__name__)

class OnlineTrainer:
    """
    🔄 Online Training System
    
    Implements online training for incremental learning and real-time
    model updates in cryptocurrency trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the online trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.scalers = {}
        self.training_history = {}
        self.performance_metrics = {}
        self.concept_drift_detector = {}
        
        # Online training parameters
        self.online_params = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'update_frequency': 100,  # Update every 100 samples
            'memory_size': 10000,  # Maximum memory size
            'forgetting_factor': 0.99,  # Exponential forgetting
            'drift_threshold': 0.1,  # Concept drift threshold
            'performance_window': 1000,  # Performance monitoring window
            'retrain_threshold': 0.05  # Retrain threshold
        }
        
        # Model configurations
        self.model_configs = {
            'sgd_regressor': {
                'loss': 'squared_loss',
                'penalty': 'l2',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'eta0': 0.01
            },
            'lightgbm_online': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            },
            'neural_online': {
                'layers': [64, 32],
                'dropout': 0.2,
                'activation': 'relu',
                'optimizer': 'adam'
            }
        }
        
        logger.info("🔄 Online Trainer initialized")
    
    def create_sgd_model(self, input_dim: int) -> SGDRegressor:
        """Create an SGD regressor for online learning."""
        try:
            config = self.model_configs['sgd_regressor']
            model = SGDRegressor(
                loss=config['loss'],
                penalty=config['penalty'],
                alpha=config['alpha'],
                learning_rate=config['learning_rate'],
                eta0=config['eta0'],
                random_state=42
            )
            
            # Initialize with dummy data
            dummy_X = np.random.randn(10, input_dim)
            dummy_y = np.random.randn(10)
            model.partial_fit(dummy_X, dummy_y)
            
            self.models['sgd_regressor'] = model
            logger.info("✅ Created SGD regressor for online learning")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create SGD model: {e}")
            return None
    
    def create_lightgbm_online(self, input_dim: int) -> lgb.LGBMRegressor:
        """Create a LightGBM model for online learning."""
        try:
            config = self.model_configs['lightgbm_online']
            model = lgb.LGBMRegressor(
                objective=config['objective'],
                metric=config['metric'],
                boosting_type=config['boosting_type'],
                num_leaves=config['num_leaves'],
                learning_rate=config['learning_rate'],
                feature_fraction=config['feature_fraction'],
                bagging_fraction=config['bagging_fraction'],
                bagging_freq=config['bagging_freq'],
                verbose=config['verbose'],
                random_state=42
            )
            
            self.models['lightgbm_online'] = model
            logger.info("✅ Created LightGBM model for online learning")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create LightGBM model: {e}")
            return None
    
    def create_neural_online(self, input_dim: int) -> Sequential:
        """Create a neural network for online learning."""
        try:
            config = self.model_configs['neural_online']
            
            model = Sequential()
            
            # Input layer
            model.add(Dense(config['layers'][0], activation=config['activation'], 
                          input_dim=input_dim))
            model.add(Dropout(config['dropout']))
            
            # Hidden layers
            for units in config['layers'][1:]:
                model.add(Dense(units, activation=config['activation']))
                model.add(Dropout(config['dropout']))
            
            # Output layer
            model.add(Dense(1))
            
            # Compile model
            optimizer = Adam(learning_rate=self.online_params['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            self.models['neural_online'] = model
            logger.info("✅ Created neural network for online learning")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create neural model: {e}")
            return None
    
    def create_all_models(self, input_dim: int):
        """Create all online learning models."""
        try:
            self.create_sgd_model(input_dim)
            self.create_lightgbm_online(input_dim)
            self.create_neural_online(input_dim)
            
            logger.info(f"✅ Created {len(self.models)} online learning models")
            
        except Exception as e:
            logger.error(f"❌ Failed to create online models: {e}")
    
    def update_model_online(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Update a model with new data using online learning."""
        try:
            model = self.models.get(model_name)
            if model is None:
                logger.error(f"❌ Model {model_name} not found")
                return {}
            
            # Scale features if needed
            if model_name not in self.scalers:
                self.scalers[model_name] = StandardScaler()
                X_scaled = self.scalers[model_name].fit_transform(X)
            else:
                X_scaled = self.scalers[model_name].transform(X)
            
            # Update model based on type
            if model_name == 'sgd_regressor':
                # SGD regressor supports partial_fit
                model.partial_fit(X_scaled, y)
                
            elif model_name == 'lightgbm_online':
                # LightGBM online learning
                model.fit(X_scaled, y, callbacks=[lgb.log_evaluation(period=0)])
                
            elif model_name == 'neural_online':
                # Neural network online learning
                model.fit(X_scaled, y, epochs=1, batch_size=self.online_params['batch_size'], 
                         verbose=0)
            
            # Calculate performance metrics
            y_pred = self.predict(model_name, X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Store training history
            if model_name not in self.training_history:
                self.training_history[model_name] = []
            
            self.training_history[model_name].append({
                'timestamp': datetime.now(),
                'samples': len(X),
                'mse': mse,
                'r2_score': r2
            })
            
            # Update performance metrics
            self.performance_metrics[model_name] = {
                'mse': mse,
                'r2_score': r2,
                'last_update': datetime.now()
            }
            
            logger.info(f"✅ Updated {model_name} - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            return {
                'model_name': model_name,
                'mse': mse,
                'r2_score': r2,
                'samples_processed': len(X)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to update {model_name}: {e}")
            return {}
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with an online model."""
        try:
            model = self.models.get(model_name)
            if model is None:
                logger.error(f"❌ Model {model_name} not found")
                return np.array([])
            
            # Scale features if scaler exists
            if model_name in self.scalers:
                X_scaled = self.scalers[model_name].transform(X)
            else:
                X_scaled = X
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to predict with {model_name}: {e}")
            return np.array([])
    
    def detect_concept_drift(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Detect concept drift in the data."""
        try:
            drift_analysis = {
                'drift_detected': False,
                'drift_score': 0.0,
                'confidence': 0.0,
                'recommendations': []
            }
            
            # Get recent performance
            if model_name in self.training_history:
                recent_history = self.training_history[model_name][-self.online_params['performance_window']:]
                
                if len(recent_history) > 10:
                    # Calculate performance trend
                    recent_mse = [entry['mse'] for entry in recent_history]
                    mse_trend = np.polyfit(range(len(recent_mse)), recent_mse, 1)[0]
                    
                    # Calculate performance volatility
                    mse_volatility = np.std(recent_mse)
                    
                    # Detect drift based on performance degradation
                    if mse_trend > 0 and mse_volatility > self.online_params['drift_threshold']:
                        drift_analysis['drift_detected'] = True
                        drift_analysis['drift_score'] = mse_trend
                        drift_analysis['confidence'] = min(1.0, mse_volatility)
                        drift_analysis['recommendations'].append("Performance degradation detected - consider retraining")
                    
                    # Detect drift based on R² decline
                    recent_r2 = [entry['r2_score'] for entry in recent_history]
                    r2_trend = np.polyfit(range(len(recent_r2)), recent_r2, 1)[0]
                    
                    if r2_trend < -self.online_params['drift_threshold']:
                        drift_analysis['drift_detected'] = True
                        drift_analysis['drift_score'] = abs(r2_trend)
                        drift_analysis['confidence'] = max(drift_analysis['confidence'], abs(r2_trend))
                        drift_analysis['recommendations'].append("R² decline detected - model may need updating")
            
            # Store drift analysis
            self.concept_drift_detector[model_name] = drift_analysis
            
            return drift_analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to detect concept drift: {e}")
            return {'drift_detected': False, 'drift_score': 0.0, 'confidence': 0.0, 'recommendations': []}
    
    def adaptive_learning_rate(self, model_name: str, performance_history: List[Dict[str, Any]]) -> float:
        """Adapt learning rate based on performance."""
        try:
            if len(performance_history) < 10:
                return self.online_params['learning_rate']
            
            # Calculate recent performance trend
            recent_mse = [entry['mse'] for entry in performance_history[-10:]]
            mse_trend = np.polyfit(range(len(recent_mse)), recent_mse, 1)[0]
            
            # Adjust learning rate based on trend
            if mse_trend > 0:  # Performance degrading
                new_lr = self.online_params['learning_rate'] * 1.1  # Increase learning rate
            elif mse_trend < -0.001:  # Performance improving
                new_lr = self.online_params['learning_rate'] * 0.95  # Decrease learning rate
            else:
                new_lr = self.online_params['learning_rate']
            
            # Ensure learning rate is within bounds
            new_lr = max(1e-6, min(new_lr, 0.1))
            
            return new_lr
            
        except Exception as e:
            logger.error(f"❌ Failed to adapt learning rate: {e}")
            return self.online_params['learning_rate']
    
    def incremental_feature_selection(self, model_name: str, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Perform incremental feature selection."""
        try:
            # Simple feature importance based on correlation
            feature_importance = []
            
            for i in range(X.shape[1]):
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                feature_importance.append(abs(correlation))
            
            # Select top features
            threshold = np.percentile(feature_importance, 75)  # Top 25% features
            selected_features = [i for i, importance in enumerate(feature_importance) if importance > threshold]
            
            return selected_features
            
        except Exception as e:
            logger.error(f"❌ Failed to perform feature selection: {e}")
            return list(range(X.shape[1]))
    
    async def continuous_online_training(self, data_stream: Any, model_names: List[str] = None):
        """Perform continuous online training from a data stream."""
        try:
            if model_names is None:
                model_names = list(self.models.keys())
            
            sample_count = 0
            batch_data = []
            
            logger.info(f"🔄 Starting continuous online training for {len(model_names)} models")
            
            async for data_point in data_stream:
                # Add to batch
                batch_data.append(data_point)
                sample_count += 1
                
                # Update models when batch is full
                if len(batch_data) >= self.online_params['batch_size']:
                    # Prepare batch
                    X_batch = np.array([point['features'] for point in batch_data])
                    y_batch = np.array([point['target'] for point in batch_data])
                    
                    # Update each model
                    for model_name in model_names:
                        # Update model
                        update_result = self.update_model_online(model_name, X_batch, y_batch)
                        
                        # Detect concept drift
                        drift_analysis = self.detect_concept_drift(model_name, X_batch, y_batch)
                        
                        if drift_analysis['drift_detected']:
                            logger.warning(f"⚠️ Concept drift detected in {model_name}: {drift_analysis['recommendations']}")
                    
                    # Clear batch
                    batch_data = []
                    
                    # Log progress
                    if sample_count % 1000 == 0:
                        logger.info(f"🔄 Processed {sample_count} samples")
            
            logger.info(f"✅ Continuous online training completed - {sample_count} samples processed")
            
        except Exception as e:
            logger.error(f"❌ Continuous online training failed: {e}")
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        try:
            if model_name not in self.training_history:
                return {}
            
            history = self.training_history[model_name]
            
            if len(history) == 0:
                return {}
            
            # Calculate performance statistics
            mse_values = [entry['mse'] for entry in history]
            r2_values = [entry['r2_score'] for entry in history]
            
            performance = {
                'current_mse': mse_values[-1],
                'current_r2': r2_values[-1],
                'avg_mse': np.mean(mse_values),
                'avg_r2': np.mean(r2_values),
                'mse_trend': np.polyfit(range(len(mse_values)), mse_values, 1)[0],
                'r2_trend': np.polyfit(range(len(r2_values)), r2_values, 1)[0],
                'total_updates': len(history),
                'last_update': history[-1]['timestamp']
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance for {model_name}: {e}")
            return {}
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare performance of all online models."""
        try:
            comparison = {}
            
            for model_name in self.models.keys():
                performance = self.get_model_performance(model_name)
                if performance:
                    comparison[model_name] = performance
            
            # Find best performing model
            if comparison:
                best_model = min(comparison.keys(), 
                               key=lambda x: comparison[x]['current_mse'])
                comparison['best_model'] = best_model
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Failed to compare models: {e}")
            return {}
    
    def get_online_trainer_summary(self) -> Dict[str, Any]:
        """Get a summary of online training activities."""
        return {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'online_params': self.online_params,
            'model_configs': list(self.model_configs.keys()),
            'performance_metrics': self.performance_metrics,
            'concept_drift_detected': any(drift['drift_detected'] for drift in self.concept_drift_detector.values())
        }
    
    def save_online_state(self, filepath: str):
        """Save online training state."""
        try:
            import pickle
            
            online_state = {
                'models': self.models,
                'scalers': self.scalers,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics,
                'concept_drift_detector': self.concept_drift_detector,
                'online_params': self.online_params
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(online_state, f)
            
            logger.info(f"💾 Online training state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save online state: {e}")
    
    def load_online_state(self, filepath: str):
        """Load online training state."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                online_state = pickle.load(f)
            
            self.models = online_state['models']
            self.scalers = online_state['scalers']
            self.training_history = online_state['training_history']
            self.performance_metrics = online_state['performance_metrics']
            self.concept_drift_detector = online_state['concept_drift_detector']
            self.online_params = online_state['online_params']
            
            logger.info(f"📂 Online training state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load online state: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'online_training_enabled': True,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    
    # Initialize online trainer
    online_trainer = OnlineTrainer(config)
    
    # Create sample data
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    
    # Create models
    online_trainer.create_all_models(10)
    
    # Simulate online training
    for i in range(0, len(X), 32):
        batch_X = X[i:i+32]
        batch_y = y[i:i+32]
        
        # Update models
        for model_name in online_trainer.models.keys():
            online_trainer.update_model_online(model_name, batch_X, batch_y)
    
    # Compare models
    comparison = online_trainer.compare_models()
    print(f"Model comparison: {comparison}")
    
    # Get trainer summary
    summary = online_trainer.get_online_trainer_summary()
    print(f"Online trainer initialized with {summary['total_models']} models") 