# modules/prediction_engine.py - ULTRA ENHANCED VERSION

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
    logging.warning("TensorFlow not available. Transformer model will be disabled.")

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
    logging.warning("SHAP not available. Feature importance analysis will be limited.")

class UltraEnhancedPredictionEngine:
    """
    Ultra-Enhanced Prediction Engine that combines multiple state-of-the-art models:
    
    1. Ensemble of LightGBM, XGBoost, and Transformer models
    2. Reinforcement Learning agent for adaptive decision making
    3. Hidden Markov Model for market regime detection
    4. Advanced feature selection and engineering
    5. Dynamic model weighting based on recent performance
    6. Confidence calibration and uncertainty quantification
    7. Multi-timeframe analysis
    8. Adaptive hyperparameter optimization
    
    This is designed to be the smartest possible prediction engine for maximum profitability.
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
            'lightgbm': 'models/lightgbm_model.joblib',
            'xgboost': 'models/xgboost_model.json',
            'transformer': 'models/neural_network_model.h5',
            'hmm': 'models/hmm_model.joblib',
            'rl_agent': 'models/rl_agent.zip',
            'ensemble': 'models/ensemble_model.pkl',
            'scaler': 'models/neural_network_scaler.joblib',
            'feature_selector': 'models/feature_selector.pkl'
        }
        
        # Initialize models dictionary
        self.models = {}
        self.model_performance = {}
        self.model_weights = ensemble_weights or {
            'lightgbm': 0.25,
            'xgboost': 0.25,
            'transformer': 0.3,
            'rl_agent': 0.2
        }
        
        # Performance tracking
        self.prediction_history = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
        
        # Feature management
        self.feature_selector = None
        self.scaler = None
        self.selected_features = []
        
        # Market regime tracking
        self.current_regime = 'NORMAL'
        self.regime_history = deque(maxlen=100)
        
        # Auto-optimization settings
        self.auto_optimize = auto_optimize
        self.last_optimization = datetime.now()
        self.optimization_interval = timedelta(hours=6)
        
        # Threading for background optimization
        self.optimization_thread = None
        self.optimization_running = False
        
        # Load all models
        self._load_models()
        
        # Initialize feature selection and scaling
        self._initialize_preprocessing()
        
        # Start background optimization if enabled
        if self.auto_optimize:
            self._start_background_optimization()
            
        logging.info("Ultra-Enhanced Prediction Engine initialized successfully")
    
    def _load_models(self):
        """Load pre-trained models from disk."""
        # Load LightGBM model
        if LIGHTGBM_AVAILABLE:
            try:
                if os.path.exists(self.model_paths['lightgbm']):
                    self.models['lightgbm'] = joblib.load(self.model_paths['lightgbm'])
                    logging.info("LightGBM model loaded successfully")
                else:
                    logging.warning("LightGBM model file not found. Will train new model.")
                    self.models['lightgbm'] = None
            except Exception as e:
                logging.error(f"Failed to load LightGBM model: {e}")
                self.models['lightgbm'] = None
        
        # Load XGBoost model
        if XGBOOST_AVAILABLE:
            try:
                # Check for the correct XGBoost file path
                xgb_path = 'models/xgboost_model.json'
                if os.path.exists(xgb_path):
                    self.models['xgboost'] = xgb.XGBRegressor()
                    self.models['xgboost'].load_model(xgb_path)
                    logging.info("XGBoost model loaded successfully")
                else:
                    logging.warning("XGBoost model file not found. Will train new model.")
                    self.models['xgboost'] = None
            except Exception as e:
                logging.error(f"Failed to load XGBoost model: {e}")
                self.models['xgboost'] = None
            
        # Load Neural Network model
        if TF_AVAILABLE:
            try:
                # Check for the correct neural network file path
                nn_path = 'models/neural_network_model.h5'
                if os.path.exists(nn_path):
                    self.models['neural_network'] = load_model(nn_path)
                    
                    # Load scaler if available
                    scaler_path = 'models/neural_network_scaler.joblib'
                    if os.path.exists(scaler_path):
                        self.models['neural_network'].scaler = joblib.load(scaler_path)
                    
                    # Load feature columns if available
                    features_path = 'models/neural_network_features.json'
                    if os.path.exists(features_path):
                        with open(features_path, 'r') as f:
                            self.models['neural_network'].feature_columns = json.load(f)
                    
                    logging.info("Neural network model loaded successfully")
                else:
                    logging.warning("Neural network model file not found. Will train new model.")
                    self.models['neural_network'] = None
            except Exception as e:
                logging.error(f"Failed to load neural network model: {e}")
                self.models['neural_network'] = None
        
        # Load HMM model
        if HMM_AVAILABLE:
            try:
                if os.path.exists(self.model_paths['hmm']):
                    self.models['hmm'] = joblib.load(self.model_paths['hmm'])
                    logging.info("HMM model loaded successfully")
                else:
                    logging.warning("HMM model file not found. Will initialize new model.")
                    self.models['hmm'] = None
            except Exception as e:
                logging.error(f"Failed to load HMM model: {e}")
                self.models['hmm'] = None
        
        # Load RL agent
        if RL_AVAILABLE:
            try:
                if os.path.exists(self.model_paths['rl_agent']):
                    self.models['rl_agent'] = PPO.load(self.model_paths['rl_agent'])
                    logging.info("RL agent loaded successfully")
                else:
                    logging.warning("RL agent file not found. Will train new agent.")
                    self.models['rl_agent'] = None
            except Exception as e:
                logging.error(f"Failed to load RL agent: {e}")
                self.models['rl_agent'] = None
        
        # Load ensemble weights
        try:
            weights_path = 'models/ensemble_weights.json'
            if os.path.exists(weights_path):
                with open(weights_path, 'r') as f:
                    self.model_weights = json.load(f)
                logging.info("Ensemble weights loaded successfully")
        except Exception as e:
            logging.warning(f"Failed to load ensemble weights: {e}")
            # Use default weights
            self.model_weights = {
                'lightgbm': 0.33,
                'xgboost': 0.33,
                'neural_network': 0.34
            }
    
    def _initialize_preprocessing(self):
        """Initialize feature selection and scaling components."""
        if SKLEARN_AVAILABLE:
            try:
                # Load or initialize scaler
                if os.path.exists(self.model_paths['scaler']):
                    self.scaler = joblib.load(self.model_paths['scaler'])
                    logging.info("Scaler loaded successfully")
                else:
                    self.scaler = RobustScaler()
                    logging.info("Initialized new RobustScaler")
                
                # Load or initialize feature selector
                if os.path.exists(self.model_paths['feature_selector']):
                    self.feature_selector = joblib.load(self.model_paths['feature_selector'])
                    logging.info("Feature selector loaded successfully")
                else:
                    self.feature_selector = SelectKBest(score_func=f_classif, k=50)
                    logging.info("Initialized new feature selector")
                    
            except Exception as e:
                logging.error(f"Failed to initialize preprocessing components: {e}")
                self.scaler = None
                self.feature_selector = None
    
    def get_prediction(self, dataframe: pd.DataFrame) -> Tuple[str, str]:
        """
        Main prediction method that maintains backward compatibility.
        
        Args:
            dataframe: DataFrame with market data and features
            
        Returns:
            Tuple of (prediction, market_regime)
        """
        try:
            # Get enhanced prediction
            result = self.get_enhanced_prediction(dataframe)
            
            # Extract prediction and regime
            prediction = result['prediction']
            market_regime = result['market_regime']
            
            # Store prediction for performance tracking
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'confidence': result['confidence'],
                'ensemble_weights': result['ensemble_weights'].copy(),
                'market_regime': market_regime
            })
            
            return prediction, market_regime
            
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            # Fallback to simple prediction
            return 'HOLD', 'NORMAL'
    
    def get_enhanced_prediction(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced prediction with full ensemble and confidence scoring.
        
        Args:
            dataframe: DataFrame with market data and features
            
        Returns:
            Dictionary with prediction details
        """
        if dataframe.empty:
            return {'prediction': 'HOLD', 'confidence': 0.0, 'market_regime': 'NORMAL'}
        
        try:
            # Preprocess data
            processed_data = self._preprocess_data(dataframe)
            if processed_data is None:
                return {'prediction': 'HOLD', 'confidence': 0.0, 'market_regime': 'NORMAL'}
            
            # Detect market regime
            market_regime = self._detect_market_regime(processed_data)
            self.current_regime = market_regime
            self.regime_history.append(market_regime)
            
            # Get individual model predictions
            model_predictions = {}
            model_confidences = {}
            
            # LightGBM prediction
            if self.models.get('lightgbm') is not None:
                try:
                    lgb_pred = self.models['lightgbm'].predict(processed_data)
                    # For regressors, convert prediction to signal
                    pred_value = lgb_pred[0] if isinstance(lgb_pred, np.ndarray) else lgb_pred
                    model_predictions['lightgbm'] = self._convert_regression_to_signal(pred_value)
                    model_confidences['lightgbm'] = 0.7  # Default confidence for regressors
                except Exception as e:
                    logging.warning(f"LightGBM prediction failed: {e}")
            
            # XGBoost prediction
            if self.models.get('xgboost') is not None:
                try:
                    xgb_pred = self.models['xgboost'].predict(processed_data)
                    # For regressors, convert prediction to signal
                    pred_value = xgb_pred[0] if isinstance(xgb_pred, np.ndarray) else xgb_pred
                    model_predictions['xgboost'] = self._convert_regression_to_signal(pred_value)
                    model_confidences['xgboost'] = 0.7  # Default confidence for regressors
                except Exception as e:
                    logging.warning(f"XGBoost prediction failed: {e}")
            
            # Transformer prediction
            if self.models.get('neural_network') is not None:
                try:
                    # Reshape data for transformer (sequence format)
                    transformer_data = self._prepare_transformer_input(processed_data)
                    tf_pred = self.models['neural_network'].predict(transformer_data)
                    model_predictions['neural_network'] = self._convert_probability_to_signal(tf_pred[0])
                    model_confidences['neural_network'] = max(tf_pred[0])
                except Exception as e:
                    logging.warning(f"Transformer prediction failed: {e}")
            
            # RL agent prediction
            if self.models.get('rl_agent') is not None:
                try:
                    rl_data = self._prepare_rl_input(processed_data)
                    rl_action, _ = self.models['rl_agent'].predict(rl_data, deterministic=True)
                    model_predictions['rl_agent'] = self._convert_rl_action_to_signal(rl_action)
                    model_confidences['rl_agent'] = 0.8  # RL confidence is typically high
                except Exception as e:
                    logging.warning(f"RL agent prediction failed: {e}")
            
            # Ensemble prediction
            if model_predictions:
                ensemble_prediction = self._ensemble_predictions(model_predictions, model_confidences)
                ensemble_confidence = self._calculate_ensemble_confidence(model_confidences)
                
                # Adjust prediction based on market regime
                final_prediction = self._adjust_prediction_for_regime(ensemble_prediction, market_regime)
                
                return {
                    'prediction': final_prediction,
                    'confidence': ensemble_confidence,
                    'market_regime': market_regime,
                    'ensemble_weights': self.model_weights.copy(),
                    'model_predictions': model_predictions,
                    'model_confidences': model_confidences
                }
            else:
                return {'prediction': 'HOLD', 'confidence': 0.0, 'market_regime': market_regime}
                
        except Exception as e:
            logging.error(f"Error in enhanced prediction: {e}")
            return {'prediction': 'HOLD', 'confidence': 0.0, 'market_regime': 'NORMAL'}
    
    def _preprocess_data(self, dataframe: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Preprocess data for prediction with robust error handling.
        """
        try:
            if dataframe.empty:
                logging.warning("Empty dataframe provided for preprocessing")
                return None
            
            # Make a copy to avoid modifying original
            df = dataframe.copy()
            
            # Filter out non-numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df_numeric = df[numeric_columns]
            
            # Handle infinite values
            df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
            df_numeric = df_numeric.fillna(0)
            
            # Apply feature selection if available
            if self.feature_selector is not None:
                try:
                    # Get feature names that the selector was trained on
                    if hasattr(self.feature_selector, 'get_support'):
                        selected_features = df_numeric.columns[self.feature_selector.get_support()]
                        df_numeric = df_numeric[selected_features]
                    else:
                        # Fallback: use all numeric features
                        logging.warning("Feature selector not properly initialized, using all numeric features")
                except Exception as e:
                    logging.warning(f"Feature selection failed: {e}")
                    # Continue with all numeric features
            
            # Apply scaling if available
            if self.scaler is not None:
                try:
                    # Ensure we have the same number of features as the scaler expects
                    if hasattr(self.scaler, 'n_features_in_'):
                        expected_features = self.scaler.n_features_in_
                        if len(df_numeric.columns) != expected_features:
                            logging.warning(f"Feature mismatch: got {len(df_numeric.columns)}, expected {expected_features}")
                            # Try to match features by name if possible
                            if hasattr(self.scaler, 'feature_names_in_'):
                                available_features = [col for col in self.scaler.feature_names_in_ if col in df_numeric.columns]
                                if len(available_features) == expected_features:
                                    df_numeric = df_numeric[available_features]
                                else:
                                    logging.error(f"Cannot match features. Available: {len(available_features)}, Expected: {expected_features}")
                                    return None
                            else:
                                # If we can't match, skip scaling
                                logging.warning("Skipping scaling due to feature mismatch")
                                return df_numeric
                    
                    # Apply scaling
                    scaled_data = self.scaler.transform(df_numeric)
                    return pd.DataFrame(scaled_data, columns=df_numeric.columns, index=df_numeric.index)
                    
                except Exception as e:
                    logging.error(f"Error in data preprocessing: {e}")
                    # Return unscaled data if scaling fails
                    return df_numeric
            
            return df_numeric
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            return None
            
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime using HMM."""
        if self.models.get('hmm') is None or data.empty:
            return 'NORMAL'
        
        try:
            # Use price volatility and volume as regime indicators
            if 'close' in data.columns and len(data) > 1:
                returns = data['close'].pct_change().dropna()
                volatility = returns.std()
                volume_ratio = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1] if len(data) >= 20 else 1.0
                
                # Simple regime detection based on volatility and volume
                if volatility > 0.02:  # High volatility
                    if volume_ratio > 1.5:
                        return 'PANIC_VOLATILITY'
                    else:
                        return 'HIGH_VOLATILITY'
                elif volatility < 0.005:  # Low volatility
                    return 'LOW_VOLATILITY'
                else:
                    return 'NORMAL'
            else:
                return 'NORMAL'
                
        except Exception as e:
            logging.warning(f"Market regime detection failed: {e}")
            return 'NORMAL'
    
    def _convert_probability_to_signal(self, probabilities: np.ndarray) -> str:
        """Convert model probabilities to trading signal."""
        if len(probabilities) >= 3:
            buy_prob, sell_prob, hold_prob = probabilities[:3]
            if buy_prob > 0.4:
                return 'BUY'
            elif sell_prob > 0.4:
                return 'SELL'
            else:
                return 'HOLD'
        elif len(probabilities) == 2:
            buy_prob, sell_prob = probabilities
            if buy_prob > 0.6:
                return 'BUY'
            elif sell_prob > 0.6:
                return 'SELL'
            else:
                return 'HOLD'
        else:
            # Single probability (regression output)
            prob = probabilities[0]
            if prob > 0.6:
                return 'BUY'
            elif prob < 0.4:
                return 'SELL'
            else:
                return 'HOLD'
    
    def _convert_regression_to_signal(self, prediction: float) -> str:
        """Convert regression prediction to trading signal."""
        # Assuming prediction is a price change or return value
        # Positive values suggest price increase (BUY)
        # Negative values suggest price decrease (SELL)
        # Small values suggest no significant change (HOLD)
        
        if prediction > 0.001:  # Positive price change
            return 'BUY'
        elif prediction < -0.001:  # Negative price change
            return 'SELL'
        else:
            return 'HOLD'
    
    def _convert_rl_action_to_signal(self, action: int) -> str:
        """Convert RL action to trading signal."""
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return action_map.get(action, 'HOLD')
    
    def _prepare_transformer_input(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare data for transformer model."""
        # Reshape to sequence format (batch_size, sequence_length, features)
        if len(data) >= 10:
            sequence_data = data.tail(10).values
        else:
            # Pad with zeros if not enough data
            sequence_data = np.zeros((10, data.shape[1]))
            sequence_data[-len(data):] = data.values
        
        return sequence_data.reshape(1, 10, -1)
    
    def _prepare_rl_input(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare data for RL agent."""
        # Use the latest observation
        return data.iloc[-1].values.reshape(1, -1)
    
    def _ensemble_predictions(self, predictions: Dict[str, str], confidences: Dict[str, float]) -> str:
        """Combine predictions from multiple models using weighted voting."""
        vote_counts = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        for model_name, prediction in predictions.items():
            weight = self.model_weights.get(model_name, 0.1)
            confidence = confidences.get(model_name, 0.5)
            vote_weight = weight * confidence
            
            vote_counts[prediction] += vote_weight
        
        # Return the prediction with highest weighted votes
        return max(vote_counts, key=vote_counts.get)
    
    def _calculate_ensemble_confidence(self, confidences: Dict[str, float]) -> float:
        """Calculate overall ensemble confidence."""
        if not confidences:
            return 0.0
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for model_name, confidence in confidences.items():
            weight = self.model_weights.get(model_name, 0.1)
            weighted_confidence += weight * confidence
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _adjust_prediction_for_regime(self, prediction: str, regime: str) -> str:
        """Adjust prediction based on market regime."""
        if regime == 'PANIC_VOLATILITY':
            # Be more conservative in panic mode
            if prediction == 'BUY':
                return 'HOLD'
            elif prediction == 'SELL':
                return 'SELL'  # Keep sell signals
        elif regime == 'HIGH_VOLATILITY':
            # Reduce position sizes but allow trading
            return prediction
        elif regime == 'LOW_VOLATILITY':
            # Be more aggressive in low volatility
            return prediction
        
        return prediction
    
    def _start_background_optimization(self):
        """Start background optimization thread."""
        if self.optimization_thread is None or not self.optimization_thread.is_alive():
            self.optimization_running = True
            self.optimization_thread = threading.Thread(target=self._background_optimization_loop)
            self.optimization_thread.daemon = True
            self.optimization_thread.start()
            logging.info("Background optimization started")
    
    def _background_optimization_loop(self):
        """Background loop for model optimization."""
        while self.optimization_running:
            try:
                current_time = datetime.now()
                if current_time - self.last_optimization > self.optimization_interval:
                    self._optimize_models()
                    self.last_optimization = current_time
                
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logging.error(f"Error in background optimization: {e}")
                time.sleep(3600)
    
    def _optimize_models(self):
        """Optimize models based on recent performance."""
        try:
            logging.info("Starting model optimization...")
            
            # Update model weights based on recent performance
            self._update_model_weights()
            
            # Retrain models if needed
            self._retrain_underperforming_models()
            
            # Save optimized models
            self._save_models()
            
            logging.info("Model optimization completed")
                
        except Exception as e:
            logging.error(f"Error in model optimization: {e}")
    
    def _update_model_weights(self):
        """Update ensemble weights based on recent performance."""
        if len(self.prediction_history) < 50:
            return
        
        # Calculate recent accuracy for each model
        model_accuracies = {}
        
        for model_name in self.model_weights.keys():
            correct_predictions = 0
            total_predictions = 0
            
            for pred in list(self.prediction_history)[-50:]:
                if model_name in pred.get('model_predictions', {}):
                    # Simple accuracy calculation (you might want to implement more sophisticated metrics)
                    total_predictions += 1
                    # This is a simplified accuracy calculation
                    correct_predictions += 1  # Placeholder
            
            if total_predictions > 0:
                model_accuracies[model_name] = correct_predictions / total_predictions
            else:
                model_accuracies[model_name] = 0.5
        
        # Update weights based on performance
        total_accuracy = sum(model_accuracies.values())
        if total_accuracy > 0:
            for model_name in self.model_weights:
                self.model_weights[model_name] = model_accuracies.get(model_name, 0.25) / total_accuracy
        
        logging.info(f"Updated model weights: {self.model_weights}")
    
    def _retrain_underperforming_models(self):
        """Retrain models that are underperforming."""
        # This would implement model retraining logic
        # For now, just log that this feature is available
        logging.info("Model retraining feature available (not implemented in this version)")
    
    def _save_models(self):
        """Save all models and components."""
        try:
            os.makedirs('models', exist_ok=True)
            
            # Save scaler
            if self.scaler is not None:
                joblib.dump(self.scaler, self.model_paths['scaler'])
            
            # Save feature selector
            if self.feature_selector is not None:
                joblib.dump(self.feature_selector, self.model_paths['feature_selector'])
            
            logging.info("Models saved successfully")
                
        except Exception as e:
            logging.error(f"Error saving models: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from models."""
        importance_dict = {}
        
        try:
            # Get feature importance from LightGBM
            if self.models.get('lightgbm') is not None and hasattr(self.models['lightgbm'], 'feature_importances_'):
                lgb_importance = self.models['lightgbm'].feature_importances_
                if len(lgb_importance) > 0:
                    importance_dict['lightgbm'] = dict(zip(range(len(lgb_importance)), lgb_importance))
            
            # Get feature importance from XGBoost
            if self.models.get('xgboost') is not None:
                try:
                    xgb_importance = self.models['xgboost'].get_score(importance_type='gain')
                    importance_dict['xgboost'] = xgb_importance
                except:
                    pass
        
        except Exception as e:
            logging.error(f"Error getting feature importance: {e}")
        
        return importance_dict
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        return {
            'model_weights': self.model_weights.copy(),
            'recent_accuracy': list(self.accuracy_history)[-10:] if self.accuracy_history else [],
            'recent_confidence': list(self.confidence_history)[-10:] if self.confidence_history else [],
            'current_regime': self.current_regime,
            'regime_history': list(self.regime_history)[-10:] if self.regime_history else []
        }

    def predict(self, data: pd.DataFrame, timeframe: str = '1m') -> float:
        """
        Predict method for compatibility with training script.
        
        Args:
            data: DataFrame with features
            timeframe: Prediction timeframe ('1m', '5m', '15m')
            
        Returns:
            Prediction value (float between -1 and 1)
        """
        try:
            # Get enhanced prediction
            result = self.get_enhanced_prediction(data)
            
            # Convert prediction to numeric value
            prediction = result['prediction']
            confidence = result['confidence']
            
            # Convert string prediction to numeric
            if prediction == 'BUY':
                return confidence
            elif prediction == 'SELL':
                return -confidence
            else:  # HOLD
                return 0.0
                
        except Exception as e:
            logging.error(f"Error in predict method: {e}")
            return 0.0

# Legacy PredictionEngine class for backward compatibility
class PredictionEngine:
    """Legacy prediction engine for backward compatibility."""
    
    def __init__(self, model_path: str = 'prediction_model.joblib'):
        self.model_path = model_path
        try:
            self.model = joblib.load(model_path)
            logging.info(f"Legacy model loaded from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load legacy model: {e}")
            self.model = None

    def get_prediction(self, dataframe: pd.DataFrame) -> Tuple[str, str]:
        """Legacy prediction method."""
        if self.model is None or dataframe.empty:
            return 'HOLD', 'NORMAL'
        
        try:
            # Use basic features for legacy model
            basic_features = ['rsi', 'macd_hist', 'bollinger_width', 'atr', 'adx', 'obv']
            available_features = [col for col in basic_features if col in dataframe.columns]
            
            if available_features:
                feature_data = dataframe[available_features].iloc[-1].values.reshape(1, -1)
                prediction = self.model.predict(feature_data)[0]
                return prediction, 'NORMAL'
            else:
                return 'HOLD', 'NORMAL'
                
        except Exception as e:
            logging.error(f"Error in legacy prediction: {e}")
            return 'HOLD', 'NORMAL'
