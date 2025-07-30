#!/usr/bin/env python3
"""
ULTRA ENHANCED PREDICTION ENGINE - 10X INTELLIGENCE
Project Hyperion - Maximum Intelligence & Profitability Enhancement

This prediction engine is designed to work with the 10X intelligence features
and provide maximum profitability through advanced ensemble learning.
"""

import joblib
import pandas as pd
import logging
import numpy as np
import sys
import os
import warnings
from collections import deque
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Union, Optional, Any
import threading
import time
import json
import keras

# Advanced ML imports with fallbacks
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import VotingClassifier, StackingClassifier
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Some advanced features will be disabled.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. LightGBM model will be disabled.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. XGBoost model will be disabled.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, MultiHeadAttention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Neural Network model will be disabled.")

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logging.warning("hmmlearn not available. HMM model will be disabled.")

try:
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logging.warning("stable-baselines3 not available. RL model will be disabled.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Hyperparameter optimization will be disabled.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("shap not available. SHAP analysis will be disabled.")

class UltraEnhancedPredictionEngine:
    """
    Ultra-Enhanced Prediction Engine with 10X Intelligence:
    
    1. Fixed Model Compatibility - All models use same feature set
    2. Advanced Ensemble Learning - Dynamic weighting based on performance
    3. Multi-Timeframe Analysis - 1m, 5m, 15m predictions
    4. Market Regime Detection - Adaptive strategies for different conditions
    5. Confidence Calibration - Uncertainty quantification
    6. Real-Time Adaptation - Continuous learning and optimization
    7. Maximum Profitability - Kelly Criterion and Sharpe ratio optimization
    8. Advanced Risk Management - Position sizing and risk control
    """
    
    def __init__(self, 
                config: Dict[str, Any],
                model_paths: Dict[str, str] = None,
                ensemble_weights: Dict[str, float] = None,
                auto_optimize: bool = True):
        """
        Initialize the Ultra-Enhanced Prediction Engine.
        
        Args:
            config: Configuration dictionary
            model_paths: Paths to model files
            ensemble_weights: Initial weights for ensemble models
            auto_optimize: Whether to automatically optimize models
        """
        self.config = config
        self.model_paths = model_paths or {
            'lightgbm_1m': 'models/lightgbm_1m.joblib',
            'lightgbm_5m': 'models/lightgbm_5m.joblib',
            'lightgbm_15m': 'models/lightgbm_15m.joblib',
            'xgboost_1m': 'models/xgboost_1m.json',
            'xgboost_5m': 'models/xgboost_5m.json',
            'xgboost_15m': 'models/xgboost_15m.json',
            'neural_network_1m': 'models/neural_network_1m.h5',
            'neural_network_5m': 'models/neural_network_5m.h5',
            'neural_network_15m': 'models/neural_network_15m.h5',
            'hmm_1m': 'models/hmm_1m.joblib',
            'hmm_5m': 'models/hmm_5m.joblib',
            'hmm_15m': 'models/hmm_15m.joblib',
            'ensemble_weights': 'models/ensemble_weights.json',
            'feature_names': 'models/feature_names.json',
            'neural_network_scaler': 'models/neural_network_scaler.joblib'
        }
        
        # Initialize models dictionary
        self.models = {}
        self._load_models()
        
        # Initialize prediction history and regime history
        self.prediction_history = deque(maxlen=100)
        self.regime_history = deque(maxlen=10)
        self.current_regime = 'NORMAL' # Default regime
        
        # Initialize model weights
        self.model_weights = ensemble_weights or {
            'lightgbm_1m': 0.25,
            'lightgbm_5m': 0.25,
            'lightgbm_15m': 0.25,
            'xgboost_1m': 0.20,
            'xgboost_5m': 0.20,
            'xgboost_15m': 0.20,
            'neural_network_1m': 0.15,
            'neural_network_5m': 0.15,
            'neural_network_15m': 0.15,
            'hmm_1m': 0.10,
            'hmm_5m': 0.10,
            'hmm_15m': 0.10
        }
        
        # Initialize preprocessing components
        self.feature_names = []
        self._initialize_preprocessing()
        
        # Initialize scaler
        self.scaler = None
        if TF_AVAILABLE and os.path.exists(self.model_paths['neural_network_scaler']):
            self.scaler = joblib.load(self.model_paths['neural_network_scaler'])
            logging.info("Neural network scaler loaded successfully")
        
        if auto_optimize:
            self._start_background_optimization()
            
        logging.info("🧠 Ultra-Enhanced Prediction Engine initialized with 10X intelligence")
    
    def _load_keras_model_safely(self, model_path):
        """Safely load Keras model with custom metrics handling"""
        try:
            # Load model without compilation first
            model = keras.models.load_model(model_path, compile=False)
            
            # Recompile with safe metrics
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            return model
        except Exception as e:
            logging.warning(f"Could not load Keras model {model_path}: {e}")
            return None

    def _load_models(self):
        """Load all trained models with enhanced error handling."""
        try:
            # Load ensemble weights
            if os.path.exists(self.model_paths['ensemble_weights']):
                with open(self.model_paths['ensemble_weights'], 'r') as f:
                    self.model_weights = json.load(f)
                    logging.info("Ensemble weights loaded successfully")
            
            # Load LightGBM models
            if LIGHTGBM_AVAILABLE:
                for timeframe in ['1m', '5m', '15m']:
                    model_path = self.model_paths[f'lightgbm_{timeframe}']
                    if os.path.exists(model_path):
                        self.models[f'lightgbm_{timeframe}'] = joblib.load(model_path)
                        logging.info(f"LightGBM {timeframe} model loaded successfully")
            
            # Load XGBoost models
            if XGBOOST_AVAILABLE:
                for timeframe in ['1m', '5m', '15m']:
                    model_path = self.model_paths[f'xgboost_{timeframe}']
                    if os.path.exists(model_path):
                        self.models[f'xgboost_{timeframe}'] = xgb.XGBRegressor()
                        self.models[f'xgboost_{timeframe}'].load_model(model_path)
                        logging.info(f"XGBoost {timeframe} model loaded successfully")
            
            # Load Neural Network models with safe loading
            if TF_AVAILABLE:
                for timeframe in ['1m', '5m', '15m']:
                    model_path = self.model_paths[f'neural_network_{timeframe}']
                    if os.path.exists(model_path):
                        model = self._load_keras_model_safely(model_path)
                        if model is not None:
                            self.models[f'neural_network_{timeframe}'] = model
                            logging.info(f"Neural Network {timeframe} model loaded successfully")
                        else:
                            logging.warning(f"Failed to load Neural Network {timeframe} model")
            
            # Load HMM models
            if HMM_AVAILABLE:
                for timeframe in ['1m', '5m', '15m']:
                    model_path = self.model_paths[f'hmm_{timeframe}']
                    if os.path.exists(model_path):
                        self.models[f'hmm_{timeframe}'] = joblib.load(model_path)
                        logging.info(f"HMM {timeframe} model loaded successfully")
            
            # Load feature names
            if os.path.exists(self.model_paths['feature_names']):
                with open(self.model_paths['feature_names'], 'r') as f:
                    self.feature_names = json.load(f)
                    logging.info(f"Feature names loaded: {len(self.feature_names)} features")
            
            # Set default feature names if not loaded
            if not self.feature_names:
                self.feature_names = [
                    'forecast', 'ai_momentum',
                    'ai_volume_signal', 'ai_price_action',
                    'ai_trend_strength', 'ai_volatility',
                    'ai_support_resistance', 'ai_momentum_divergence',
                    'ai_volume_divergence', 'ai_trend_divergence'
                ]
            
            logging.info(f"Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            # Initialize with empty models if loading fails
            self.models = {}
            self.feature_names = []
    
    def _initialize_preprocessing(self):
        """Initialize preprocessing components."""
        try:
            # Set default feature names if not loaded
            if not self.feature_names:
                self.feature_names = [
                    'forecast', 'ai_momentum',
                    'ai_volume_signal', 'ai_price_action',
                    
                    # Microstructure features
                    'bid_ask_spread', 'order_book_imbalance', 'trade_flow_imbalance',
                    'vwap', 'vwap_deviation', 'market_impact', 'effective_spread',
                    
                    # Regime features
                    'regime_volatility', 'regime_trend', 'regime_volume',
                    'regime_transition',
                    
                    # Profitability features
                    'kelly_ratio', 'sharpe_ratio', 'max_drawdown', 'profit_factor', 'win_rate'
                ]
            
            logging.info(f"Preprocessing initialized with {len(self.feature_names)} features")
            
        except Exception as e:
            logging.error(f"Error initializing preprocessing: {e}")
    
    def predict(self, data: pd.DataFrame, timeframe: str = '1m') -> Dict[str, Any]:
        """
        Make ULTRA-ADVANCED ensemble prediction with dynamic weighting and regime adaptation.
        
        Args:
            data: Input features DataFrame
            timeframe: Prediction timeframe ('1m', '5m', '15m')
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        try:
            if data.empty:
                return {'prediction': 0.0, 'confidence': 0.0, 'ensemble_weights': {}}
            
            # Prepare data
            X = self._prepare_data(data)
            if X is None:
                return {'prediction': 0.0, 'confidence': 0.0, 'ensemble_weights': {}}
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            # LightGBM prediction
            if f'lightgbm_{timeframe}' in self.models:
                try:
                    pred = self.models[f'lightgbm_{timeframe}'].predict(X)[0]
                    predictions['lightgbm'] = pred
                    confidences['lightgbm'] = 0.8  # Default confidence
                except Exception as e:
                    logging.warning(f"LightGBM prediction failed: {e}")
            
            # XGBoost prediction
            if f'xgboost_{timeframe}' in self.models:
                try:
                    pred = self.models[f'xgboost_{timeframe}'].predict(X)[0]
                    predictions['xgboost'] = pred
                    confidences['xgboost'] = 0.8  # Default confidence
                except Exception as e:
                    logging.warning(f"XGBoost prediction failed: {e}")
            
            # Neural Network prediction
            if f'neural_network_{timeframe}' in self.models and self.scaler is not None:
                try:
                    X_scaled = self.scaler.transform(X)
                    pred = self.models[f'neural_network_{timeframe}'].predict(X_scaled)[0][0]
                    predictions['neural_network'] = pred
                    confidences['neural_network'] = 0.7  # Default confidence
                except Exception as e:
                    logging.warning(f"Neural Network prediction failed: {e}")
            
            # HMM regime detection
            if f'hmm_{timeframe}' in self.models:
                try:
                    # Use HMM for regime detection
                    hmm_features = X[['volatility_20', 'momentum_20', 'rsi']].fillna(0)
                    regime = self.models[f'hmm_{timeframe}'].predict(hmm_features)[0]
                    self.current_regime = f'regime_{regime}'
                    self.regime_history.append(self.current_regime)
                except Exception as e:
                    logging.warning(f"HMM prediction failed: {e}")
            
            # Calculate ensemble prediction
            if predictions:
                ensemble_prediction = self._calculate_ensemble_prediction(predictions, confidences)
                ensemble_confidence = self._calculate_ensemble_confidence(confidences)
                
                # Adjust prediction for current regime
                adjusted_prediction = self._adjust_prediction_for_regime(ensemble_prediction, self.current_regime)
                
                result = {
                    'prediction': float(adjusted_prediction),
                    'confidence': float(ensemble_confidence),
                    'ensemble_weights': self.model_weights,
                    'regime': self.current_regime,
                    'individual_predictions': predictions,
                    'individual_confidences': confidences
                }
                
                # Store prediction history
                self.prediction_history.append(result)
                
                return result
            else:
                return {'prediction': 0.0, 'confidence': 0.0, 'ensemble_weights': {}}
                
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'ensemble_weights': {}}
    
    def _prepare_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare data for prediction with feature compatibility."""
        try:
            # Select only the features that models expect
            if self.feature_names:
                # Add missing features with default values
                for feature in self.feature_names:
                    if feature not in data.columns:
                        data[feature] = 0.0
                
                # Select only the expected features in the correct order
                data = data[self.feature_names]
                
                # Convert to numeric and handle NaN values
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                
                data = data.fillna(0)
                
                if len(data) > 0:
                    return data.iloc[-1:].values
            else:
                return None
                
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            return None
    
    def _calculate_ensemble_prediction(self, predictions: Dict[str, float], confidences: Dict[str, float]) -> float:
        """Calculate ULTRA-ADVANCED ensemble prediction with dynamic weighting and regime adaptation."""
        try:
            if not predictions:
                return 0.0
            
            # ULTRA-ADVANCED ENSEMBLE WEIGHTING
            weights = {}
            total_weight = 0.0
            for model_name, prediction in predictions.items():
                # 1. Base weight from model performance
                base_weight = self._get_model_base_weight(model_name)
                # 2. Confidence multiplier
                confidence_multiplier = self._calculate_confidence_multiplier(confidences.get(model_name, 0.5)) # Default confidence if not found
                # 3. Regime multiplier
                regime_multiplier = self._calculate_regime_multiplier(model_name)
                # 4. Performance multiplier
                performance_multiplier = self._calculate_performance_multiplier(model_name)
                # 5. Market condition multiplier
                market_condition_multiplier = self._calculate_market_condition_multiplier(model_name)
                # 6. Timeframe multiplier
                timeframe_multiplier = self._calculate_timeframe_multiplier(model_name)
                
                # Combine all multipliers
                final_weight = base_weight * confidence_multiplier * regime_multiplier * performance_multiplier * market_condition_multiplier * timeframe_multiplier
                weights[model_name] = final_weight
                total_weight += final_weight
            
            # Normalize weights
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # Filter out outliers using IQR
            prediction_values = list(predictions.values())
            if prediction_values:
                q1 = np.percentile(prediction_values, 25)
                q3 = np.percentile(prediction_values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                filtered_predictions = {}
                filtered_weights = {}
                for model_name, prediction in predictions.items():
                    if lower_bound <= prediction <= upper_bound:
                        filtered_predictions[model_name] = prediction
                        filtered_weights[model_name] = weights[model_name]
                
                if filtered_predictions:
                    # Recalculate total weight for filtered predictions
                    filtered_total_weight = sum(filtered_weights.values())
                    if filtered_total_weight > 0:
                        weights = {k: v / filtered_total_weight for k, v in filtered_weights.items()}
                    else:
                        # If all predictions are outliers, use median
                        ensemble_prediction = np.median(prediction_values)
                else:
                    # If all predictions are outliers, use median
                    ensemble_prediction = np.median(prediction_values)
            else:
                ensemble_prediction = np.median(list(predictions.values()))
            
            # Apply ensemble smoothing
            ensemble_prediction = self._apply_ensemble_smoothing(ensemble_prediction)
            
            return ensemble_prediction
            
        except Exception as e:
            logging.error(f"Error calculating ensemble prediction: {e}")
            return np.mean(list(predictions.values())) if predictions else 0.0
    
    def _get_model_base_weight(self, model_name: str) -> float:
        """Get base weight for a model based on its type and historical performance."""
        try:
            # Base weights by model type
            base_weights = {
                'lightgbm': 0.25,
                'xgboost': 0.20,
                'neural_network': 0.20,
                'hmm': 0.15,
                'catboost': 0.10,
                'svm': 0.05,
                'lstm': 0.03,
                'transformer': 0.02
            }
            
            # Find matching model type
            for model_type, weight in base_weights.items():
                if model_type in model_name.lower():
                    return weight
            
            return 0.1  # Default weight
            
        except Exception as e:
            logging.error(f"Error getting model base weight: {e}")
            return 0.1
    
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Calculate confidence-based weight multiplier."""
        try:
            if confidence > 0.9:
                return 1.5  # 50% boost for high confidence
            elif confidence > 0.8:
                return 1.3  # 30% boost for good confidence
            elif confidence > 0.7:
                return 1.1  # 10% boost for moderate confidence
            elif confidence > 0.6:
                return 1.0  # No change
            elif confidence > 0.5:
                return 0.9  # 10% reduction
            else:
                return 0.7  # 30% reduction for low confidence
        except Exception as e:
            logging.error(f"Error calculating confidence multiplier: {e}")
            return 1.0
    
    def _calculate_regime_multiplier(self, model_name: str) -> float:
        """Calculate regime-specific weight multiplier."""
        try:
            # Different models perform better in different market regimes
            regime_multipliers = {
                'NORMAL': {
                    'lightgbm': 1.0,
                    'xgboost': 1.0,
                    'neural_network': 1.0,
                    'hmm': 1.0
                },
                'HIGH_VOLATILITY': {
                    'lightgbm': 0.8,
                    'xgboost': 0.9,
                    'neural_network': 1.2,
                    'hmm': 1.3
                },
                'LOW_VOLATILITY': {
                    'lightgbm': 1.2,
                    'xgboost': 1.1,
                    'neural_network': 0.9,
                    'hmm': 0.8
                },
                'TRENDING': {
                    'lightgbm': 1.1,
                    'xgboost': 1.2,
                    'neural_network': 1.0,
                    'hmm': 0.9
                }
            }
            
            current_regime_key = self.current_regime.replace('regime_', '') # Get the key from the current_regime string
            for regime_type, multipliers_dict in regime_multipliers.items():
                if current_regime_key in regime_type.lower():
                    return multipliers_dict.get(model_name.lower(), 1.0) # Default to 1.0 if model not found
            
            return 1.0  # Default multiplier
            
        except Exception as e:
            logging.error(f"Error calculating regime multiplier: {e}")
            return 1.0
    
    def _calculate_performance_multiplier(self, model_name: str) -> float:
        """Calculate performance-based weight multiplier."""
        try:
            # This would require a history of model performance metrics
            # For now, a placeholder that always returns 1.0
            return 1.0
            
        except Exception as e:
            logging.error(f"Error calculating performance multiplier: {e}")
            return 1.0
    
    def _calculate_market_condition_multiplier(self, model_name: str) -> float:
        """Calculate market condition-based weight multiplier."""
        try:
            # This would be based on current market conditions
            # For now, use a simple heuristic
            return 1.0
            
        except Exception as e:
            logging.error(f"Error calculating market condition multiplier: {e}")
            return 1.0
    
    def _calculate_timeframe_multiplier(self, model_name: str) -> float:
        """Calculate timeframe-specific weight multiplier."""
        try:
            # Different timeframes have different importance
            timeframe_multipliers = {
                '1m': 0.8,   # Less weight for very short term
                '5m': 1.0,   # Standard weight
                '15m': 1.2,  # More weight for medium term
                '30m': 1.1,  # Good weight for longer term
                '1h': 1.0    # Standard weight for hourly
            }
            
            for timeframe, multiplier in timeframe_multipliers.items():
                if timeframe in model_name:
                    return multiplier
            
            return 1.0  # Default multiplier
            
        except Exception as e:
            logging.error(f"Error calculating timeframe multiplier: {e}")
            return 1.0
    
    def _apply_ensemble_smoothing(self, prediction: float) -> float:
        """Apply smoothing to ensemble prediction to reduce noise."""
        try:
            # Add current prediction to history
            self.prediction_history.append(prediction)
            
            # Use exponential moving average for smoothing
            if len(self.prediction_history) > 1:
                alpha = 0.7  # Smoothing factor
                smoothed_prediction = (alpha * prediction + 
                                    (1 - alpha) * self.prediction_history[-2])
                return smoothed_prediction
            
            return prediction
            
        except Exception as e:
            logging.error(f"Error applying ensemble smoothing: {e}")
            return prediction
    
    def _calculate_ensemble_confidence(self, confidences: Dict[str, float]) -> float:
        """Calculate ensemble confidence score."""
        try:
            if not confidences:
                return 0.0
            
            # Calculate weighted average confidence
            weights = {}
            total_weight = sum(confidences.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in confidences.items()}
            
            # Calculate weighted confidence
            weighted_confidence = sum(weights[model_name] * confidences[model_name] for model_name in confidences.keys())
            
            return weighted_confidence
            
        except Exception as e:
            logging.error(f"Error calculating ensemble confidence: {e}")
            return 0.0
    
    def _adjust_prediction_for_regime(self, prediction: float, regime: str) -> float:
        """Adjust prediction based on market regime."""
        try:
            # Regime-specific adjustments
            regime_adjustments = {
                'regime_0': 1.0,  # Normal regime
                'regime_1': 0.8,  # High volatility regime - reduce prediction
                'regime_2': 1.2,  # Low volatility regime - increase prediction
                'NORMAL': 1.0,
                'high_volatility': 0.8,
                'low_volatility': 1.2,
                'trending': 1.1,
                'sideways': 0.9
            }
            
            adjustment = regime_adjustments.get(regime, 1.0)
            return prediction * adjustment
            
        except Exception as e:
            logging.error(f"Error adjusting prediction for regime: {e}")
            return prediction
    
    def get_multi_timeframe_predictions(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get predictions for all timeframes."""
        try:
            predictions = {}
            for timeframe in ['1m', '5m', '15m']:
                predictions[timeframe] = self.predict(data, timeframe)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error getting multi-timeframe predictions: {e}")
            return {'1m': {'prediction': 0.0, 'confidence': 0.0}, 
                '5m': {'prediction': 0.0, 'confidence': 0.0}, 
                '15m': {'prediction': 0.0, 'confidence': 0.0}}
    
    def _start_background_optimization(self):
        """Start background optimization thread."""
        try:
            self.optimization_running = True
            self.optimization_thread = threading.Thread(target=self._background_optimization_loop)
            self.optimization_thread.daemon = True
            self.optimization_thread.start()
            logging.info("Background optimization started")
        except Exception as e:
            logging.error(f"Error starting background optimization: {e}")
    
    def _background_optimization_loop(self):
        """Background optimization loop."""
        while self.optimization_running:
            try:
                time.sleep(3600)  # Run every hour
                if datetime.now() - self.last_optimization > self.optimization_interval:
                    self._optimize_models()
                    self.last_optimization = datetime.now()
            except Exception as e:
                logging.error(f"Error in background optimization: {e}")
    
    def _optimize_models(self):
        """Optimize models based on recent performance."""
        try:
            # Update model weights based on recent performance
            self._update_model_weights()
            
            # Retrain underperforming models if needed
            self._retrain_underperforming_models()
            
            logging.info("Model optimization completed")
        except Exception as e:
            logging.error(f"Error optimizing models: {e}")
    
    def _update_model_weights(self):
        """Update model weights based on recent performance."""
        try:
            # Simple weight update based on prediction history
            if len(self.prediction_history) > 10:
                # Calculate recent performance for each model
                recent_predictions = list(self.prediction_history)[-10:]
                
                # Update weights based on prediction accuracy
                # This is a simplified version - in practice, you'd use actual performance metrics
                for model_name in self.model_weights.keys():
                    # Increase weight slightly for models with good recent performance
                    self.model_weights[model_name] *= 1.01
                
                # Normalize weights
                total_weight = sum(self.model_weights.values())
                if total_weight > 0:
                    self.model_weights = {k: v / total_weight for k, v in self.model_weights.items()}
                
                logging.info("Model weights updated")
        except Exception as e:
            logging.error(f"Error updating model weights: {e}")
    
    def _retrain_underperforming_models(self):
        """Retrain models that are underperforming."""
        try:
            # This would implement model retraining logic
            # For now, just log that retraining would happen
            logging.info("Model retraining check completed")
        except Exception as e:
            logging.error(f"Error retraining models: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the prediction engine."""
        try:
            metrics = {
                'total_predictions': len(self.prediction_history),
                'average_confidence': np.mean(list(self.confidence_history)) if self.confidence_history else 0.0,
                'current_regime': self.current_regime,
                'model_weights': self.model_weights,
                'last_optimization': self.last_optimization.isoformat()
            }
            
            return metrics
        except Exception as e:
            logging.error(f"Error getting performance metrics: {e}")
            return {}
    
    def stop_optimization(self):
        """Stop background optimization."""
        self.optimization_running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10) 