import numpy as np
import pandas as pd
import logging
import joblib
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import os

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Neural networks will be limited.")

# Meta-learning
try:
    from sklearn.ensemble import StackingClassifier
    STACKING_AVAILABLE = True
except ImportError:
    STACKING_AVAILABLE = False
    logging.warning("StackingClassifier not available. Will use manual ensemble.")
from sklearn.base import BaseEstimator, ClassifierMixin

class MetaLearner(BaseEstimator, ClassifierMixin):
    """
    Advanced meta-learner that adapts to market conditions.
    """
    
    def __init__(self, base_models: List = None, meta_model = None):
        self.base_models = base_models or []
        self.meta_model = meta_model or LogisticRegression()
        self.is_fitted = False
        
    def fit(self, X, y):
        # Train base models
        for model in self.base_models:
            model.fit(X, y)
        
        # Get base predictions
        base_predictions = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models
        ])
        
        # Train meta-model
        self.meta_model.fit(base_predictions, y)
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        base_predictions = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models
        ])
        
        meta_pred = self.meta_model.predict_proba(base_predictions)
        return meta_pred
    
    def predict(self, X):
        return self.predict_proba(X)[:, 1] > 0.5

class AdvancedEnsemble:
    """
    Advanced ensemble system with stacking, blending, meta-learning,
    and online learning capabilities.
    """
    
    def __init__(self, 
                 models_dir: str = "models/ensemble",
                 use_meta_learning: bool = True,
                 use_online_learning: bool = True,
                 ensemble_size: int = 10):
        """
        Initialize the advanced ensemble system.
        
        Args:
            models_dir: Directory to save ensemble models
            use_meta_learning: Whether to use meta-learning
            use_online_learning: Whether to use online learning
            ensemble_size: Number of models in ensemble
        """
        self.models_dir = models_dir
        self.use_meta_learning = use_meta_learning
        self.use_online_learning = use_online_learning
        self.ensemble_size = ensemble_size
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models
        self.base_models = []
        self.meta_models = []
        self.online_models = []
        self.ensemble_weights = []
        self.model_performance = {}
        self.feature_importance = {}
        
        # Initialize scalers
        self.scalers = {}
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_history = []
        
        logging.info("🚀 Advanced Ensemble System initialized")
    
    def create_base_models(self) -> List:
        """
        Create diverse base models for the ensemble.
        
        Returns:
            List of base models
        """
        models = []
        
        # Tree-based models
        models.extend([
            RandomForestClassifier(n_estimators=100, random_state=42),
            RandomForestClassifier(n_estimators=200, max_depth=10, random_state=43),
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=44),
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=45)
        ])
        
        # XGBoost models
        models.extend([
            xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=46),
            xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=47),
            xgb.XGBClassifier(n_estimators=150, subsample=0.8, colsample_bytree=0.8, random_state=48)
        ])
        
        # LightGBM models
        models.extend([
            lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=49),
            lgb.LGBMClassifier(n_estimators=200, max_depth=6, random_state=50),
            lgb.LGBMClassifier(n_estimators=150, subsample=0.8, colsample_bytree=0.8, random_state=51)
        ])
        
        # Linear models
        models.extend([
            LogisticRegression(random_state=52, max_iter=1000),
            LogisticRegression(C=0.1, random_state=53, max_iter=1000),
            LogisticRegression(C=10, random_state=54, max_iter=1000)
        ])
        
        # SVM models
        models.extend([
            SVC(probability=True, random_state=55, kernel='rbf'),
            SVC(probability=True, random_state=56, kernel='linear')
        ])
        
        # Neural networks
        if TENSORFLOW_AVAILABLE:
            models.extend([
                self._create_neural_network(units=[64, 32], dropout=0.2),
                self._create_neural_network(units=[128, 64, 32], dropout=0.3),
                self._create_neural_network(units=[256, 128, 64], dropout=0.4)
            ])
        
        # Limit to ensemble size
        return models[:self.ensemble_size]
    
    def _create_neural_network(self, units: List[int], dropout: float = 0.2):
        """Create a neural network model."""
        model = keras.Sequential()
        
        for i, unit in enumerate(units):
            if i == 0:
                model.add(keras.layers.Dense(unit, activation='relu', input_shape=(None,)))
            else:
                model.add(keras.layers.Dense(unit, activation='relu'))
            
            model.add(keras.layers.Dropout(dropout))
        
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                      validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the complete ensemble system.
        
        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction for validation
            
        Returns:
            Training results and performance metrics
        """
        try:
            logging.info("🎯 Training Advanced Ensemble System")
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Create base models
            self.base_models = self.create_base_models()
            
            # Train base models
            base_performances = []
            for i, model in enumerate(self.base_models):
                logging.info(f"Training base model {i+1}/{len(self.base_models)}")
                
                # Scale features if needed
                if hasattr(model, 'coef_') or isinstance(model, SVC):
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    self.scalers[f'model_{i}'] = scaler
                    
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                
                # Calculate performance
                performance = self._calculate_performance(y_val, y_pred)
                base_performances.append(performance)
                
                logging.info(f"Base model {i+1} performance: {performance['f1_score']:.4f}")
            
            # Create meta-models
            if self.use_meta_learning:
                self._create_meta_models(X_train, y_train, X_val, y_val)
            
            # Calculate ensemble weights
            self._calculate_ensemble_weights(base_performances)
            
            # Save models
            self._save_ensemble()
            
            # Calculate final performance
            final_performance = self._evaluate_ensemble(X_val, y_val)
            
            logging.info(f"✅ Ensemble training completed. Final F1: {final_performance['f1_score']:.4f}")
            
            return {
                'base_performances': base_performances,
                'final_performance': final_performance,
                'ensemble_weights': self.ensemble_weights
            }
            
        except Exception as e:
            logging.error(f"Error training ensemble: {e}")
            return {}
    
    def _create_meta_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Create and train meta-models."""
        try:
            # Get base predictions for training
            base_predictions_train = self._get_base_predictions(X_train)
            base_predictions_val = self._get_base_predictions(X_val)
            
            # Create different meta-models
            meta_models = [
                LogisticRegression(random_state=60, max_iter=1000),
                RandomForestClassifier(n_estimators=100, random_state=61),
                xgb.XGBClassifier(n_estimators=100, random_state=62),
                lgb.LGBMClassifier(n_estimators=100, random_state=63)
            ]
            
            for i, meta_model in enumerate(meta_models):
                meta_model.fit(base_predictions_train, y_train)
                
                # Evaluate meta-model
                y_pred = meta_model.predict(base_predictions_val)
                performance = self._calculate_performance(y_val, y_pred)
                
                self.meta_models.append({
                    'model': meta_model,
                    'performance': performance,
                    'index': i
                })
                
                logging.info(f"Meta-model {i+1} performance: {performance['f1_score']:.4f}")
            
            # Sort by performance
            self.meta_models.sort(key=lambda x: x['performance']['f1_score'], reverse=True)
            
        except Exception as e:
            logging.error(f"Error creating meta-models: {e}")
    
    def _get_base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from all base models."""
        predictions = []
        
        for i, model in enumerate(self.base_models):
            if f'model_{i}' in self.scalers:
                X_scaled = self.scalers[f'model_{i}'].transform(X)
                pred = model.predict_proba(X_scaled)[:, 1]
            else:
                pred = model.predict_proba(X)[:, 1]
            
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def _calculate_ensemble_weights(self, performances: List[Dict]) -> None:
        """Calculate optimal ensemble weights based on performance."""
        try:
            # Use F1 scores as weights
            f1_scores = [p['f1_score'] for p in performances]
            
            # Softmax to get probabilities
            exp_scores = np.exp(f1_scores)
            self.ensemble_weights = exp_scores / np.sum(exp_scores)
            
            logging.info(f"Ensemble weights calculated: {self.ensemble_weights}")
            
        except Exception as e:
            logging.error(f"Error calculating ensemble weights: {e}")
            # Equal weights as fallback
            self.ensemble_weights = np.ones(len(performances)) / len(performances)
    
    def _evaluate_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate the complete ensemble."""
        try:
            # Get base predictions
            base_predictions = self._get_base_predictions(X)
            
            # Weighted ensemble prediction
            weighted_pred = np.zeros(len(X))
            for i, weight in enumerate(self.ensemble_weights):
                weighted_pred += weight * base_predictions[:, i]
            
            # Meta-model prediction (if available)
            if self.meta_models:
                meta_pred = self.meta_models[0]['model'].predict_proba(base_predictions)[:, 1]
                
                # Combine weighted and meta predictions
                final_pred = 0.7 * weighted_pred + 0.3 * meta_pred
            else:
                final_pred = weighted_pred
            
            # Convert to binary predictions
            y_pred_binary = (final_pred > 0.5).astype(int)
            
            return self._calculate_performance(y, y_pred_binary)
            
        except Exception as e:
            logging.error(f"Error evaluating ensemble: {e}")
            return {}
    
    def _calculate_performance(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
            }
        except Exception as e:
            logging.error(f"Error calculating performance: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.5
            }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the ensemble.
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            # Get base predictions
            base_predictions = self._get_base_predictions(X)
            
            # Weighted ensemble prediction
            weighted_pred = np.zeros(len(X))
            for i, weight in enumerate(self.ensemble_weights):
                weighted_pred += weight * base_predictions[:, i]
            
            # Meta-model prediction (if available)
            if self.meta_models:
                meta_pred = self.meta_models[0]['model'].predict_proba(base_predictions)[:, 1]
                
                # Combine weighted and meta predictions
                final_pred = 0.7 * weighted_pred + 0.3 * meta_pred
            else:
                final_pred = weighted_pred
            
            # Convert to binary predictions
            y_pred_binary = (final_pred > 0.5).astype(int)
            
            return y_pred_binary, final_pred
            
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def online_update(self, X_new: pd.DataFrame, y_new: pd.Series) -> None:
        """
        Update models with new data (online learning).
        
        Args:
            X_new: New feature data
            y_new: New target data
        """
        try:
            if not self.use_online_learning:
                return
            
            logging.info("🔄 Performing online update")
            
            # Update base models
            for i, model in enumerate(self.base_models):
                if hasattr(model, 'partial_fit'):
                    # Online learning for models that support it
                    if f'model_{i}' in self.scalers:
                        X_scaled = self.scalers[f'model_{i}'].transform(X_new)
                        model.partial_fit(X_scaled, y_new, classes=[0, 1])
                    else:
                        model.partial_fit(X_new, y_new, classes=[0, 1])
            
            # Update meta-models
            if self.meta_models:
                base_predictions = self._get_base_predictions(X_new)
                for meta_info in self.meta_models:
                    if hasattr(meta_info['model'], 'partial_fit'):
                        meta_info['model'].partial_fit(base_predictions, y_new, classes=[0, 1])
            
            # Track adaptation
            self.adaptation_history.append({
                'timestamp': datetime.now().isoformat(),
                'samples': len(X_new),
                'performance': self._calculate_performance(y_new, self.predict(X_new)[0])
            })
            
            logging.info("✅ Online update completed")
            
        except Exception as e:
            logging.error(f"Error in online update: {e}")
    
    def _save_ensemble(self) -> None:
        """Save the ensemble models and metadata."""
        try:
            # Save base models
            for i, model in enumerate(self.base_models):
                model_path = os.path.join(self.models_dir, f'base_model_{i}.joblib')
                joblib.dump(model, model_path)
            
            # Save meta-models
            for i, meta_info in enumerate(self.meta_models):
                model_path = os.path.join(self.models_dir, f'meta_model_{i}.joblib')
                joblib.dump(meta_info['model'], model_path)
            
            # Save scalers
            scaler_path = os.path.join(self.models_dir, 'scalers.joblib')
            joblib.dump(self.scalers, scaler_path)
            
            # Save metadata
            metadata = {
                'ensemble_weights': self.ensemble_weights.tolist(),
                'model_performance': self.model_performance,
                'performance_history': self.performance_history,
                'adaptation_history': self.adaptation_history,
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.models_dir, 'ensemble_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logging.info("💾 Ensemble models saved")
            
        except Exception as e:
            logging.error(f"Error saving ensemble: {e}")
    
    def load_ensemble(self) -> bool:
        """Load the ensemble models and metadata."""
        try:
            # Load base models
            self.base_models = []
            i = 0
            while True:
                model_path = os.path.join(self.models_dir, f'base_model_{i}.joblib')
                if not os.path.exists(model_path):
                    break
                model = joblib.load(model_path)
                self.base_models.append(model)
                i += 1
            
            # Load meta-models
            self.meta_models = []
            i = 0
            while True:
                model_path = os.path.join(self.models_dir, f'meta_model_{i}.joblib')
                if not os.path.exists(model_path):
                    break
                model = joblib.load(model_path)
                self.meta_models.append({'model': model, 'index': i})
                i += 1
            
            # Load scalers
            scaler_path = os.path.join(self.models_dir, 'scalers.joblib')
            if os.path.exists(scaler_path):
                self.scalers = joblib.load(scaler_path)
            
            # Load metadata
            metadata_path = os.path.join(self.models_dir, 'ensemble_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.ensemble_weights = np.array(metadata.get('ensemble_weights', []))
                    self.model_performance = metadata.get('model_performance', {})
                    self.performance_history = metadata.get('performance_history', [])
                    self.adaptation_history = metadata.get('adaptation_history', [])
            
            logging.info(f"📂 Loaded {len(self.base_models)} base models and {len(self.meta_models)} meta-models")
            return True
            
        except Exception as e:
            logging.error(f"Error loading ensemble: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the ensemble."""
        try:
            importance_dict = {}
            
            # Get importance from tree-based models
            for i, model in enumerate(self.base_models):
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for j, imp in enumerate(importances):
                        feature_name = f'feature_{j}'
                        if feature_name not in importance_dict:
                            importance_dict[feature_name] = []
                        importance_dict[feature_name].append(imp)
            
            # Average importance across models
            avg_importance = {}
            for feature, importances in importance_dict.items():
                avg_importance[feature] = np.mean(importances)
            
            # Sort by importance
            sorted_importance = dict(sorted(avg_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logging.error(f"Error getting feature importance: {e}")
            return {} 