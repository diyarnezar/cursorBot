"""
Maximum Intelligence Model Training
==================================

Part 2: Advanced Model Training & Optimization
Focus: Maximum Performance Over Speed

This module implements the smartest possible model training that:
- Uses advanced architectures for maximum intelligence
- Implements sophisticated ensemble methods
- Prioritizes performance over training speed
- Uses advanced hyperparameter optimization
- Implements meta-learning for continuous improvement
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Attention, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MaximumIntelligenceModelTrainer:
    """
    Maximum Intelligence Model Trainer
    Focus: Train the smartest possible models for maximum trading performance
    """
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.ensemble_weights = {}
        self.hyperparameters = {}
        self.training_history = {}
        
    def train_maximum_intelligence_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train models with maximum intelligence focus
        Priority: Best performance, not fastest training
        """
        logger.info("🧠 Training Maximum Intelligence Models...")
        
        # 1. Advanced LightGBM (Best tree-based model)
        lightgbm_model, lightgbm_score = self._train_advanced_lightgbm(X, y)
        
        # 2. Advanced XGBoost (Second best tree-based)
        xgboost_model, xgboost_score = self._train_advanced_xgboost(X, y)
        
        # 3. Advanced CatBoost (Third best tree-based)
        catboost_model, catboost_score = self._train_advanced_catboost(X, y)
        
        # 4. Advanced Neural Network (Deep learning)
        neural_model, neural_score = self._train_advanced_neural_network(X, y)
        
        # 5. Advanced LSTM (Time series specific)
        lstm_model, lstm_score = self._train_advanced_lstm(X, y)
        
        # 6. Advanced Random Forest (Ensemble)
        rf_model, rf_score = self._train_advanced_random_forest(X, y)
        
        # 7. Advanced Gradient Boosting (Ensemble)
        gb_model, gb_score = self._train_advanced_gradient_boosting(X, y)
        
        # Store models and scores
        self.models = {
            'lightgbm': lightgbm_model,
            'xgboost': xgboost_model,
            'catboost': catboost_model,
            'neural_network': neural_model,
            'lstm': lstm_model,
            'random_forest': rf_model,
            'gradient_boosting': gb_model
        }
        
        self.model_performance = {
            'lightgbm': lightgbm_score,
            'xgboost': xgboost_score,
            'catboost': catboost_score,
            'neural_network': neural_score,
            'lstm': lstm_score,
            'random_forest': rf_score,
            'gradient_boosting': gb_score
        }
        
        # 8. Create intelligent ensemble
        ensemble_weights = self._create_intelligent_ensemble()
        
        logger.info("🧠 Maximum Intelligence Models Training Complete!")
        logger.info(f"   • Best model: {max(self.model_performance, key=self.model_performance.get)} ({max(self.model_performance.values()):.3f})")
        logger.info(f"   • Average score: {np.mean(list(self.model_performance.values())):.3f}")
        
        return {
            'models': self.models,
            'performance': self.model_performance,
            'ensemble_weights': ensemble_weights
        }
    
    def _train_advanced_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
        """Train advanced LightGBM with sophisticated hyperparameter optimization"""
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-2, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'early_stopping_rounds': 50,
                'verbose': -1,
                'random_state': 42
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'early_stopping_rounds': 50,
            'verbose': -1,
            'random_state': 42
        })
        
        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(X, y)
        
        # Calculate final score
        y_pred = final_model.predict(X)
        final_score = r2_score(y, y_pred)
        
        self.hyperparameters['lightgbm'] = best_params
        logger.info(f"🧠 LightGBM trained - Score: {final_score:.3f}")
        
        return final_model, final_score
    
    def _train_advanced_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
        """Train advanced XGBoost with sophisticated hyperparameter optimization"""
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)],
                         early_stopping_rounds=50,
                         verbose=0)
                
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42
        })
        
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(X, y)
        
        # Calculate final score
        y_pred = final_model.predict(X)
        final_score = r2_score(y, y_pred)
        
        self.hyperparameters['xgboost'] = best_params
        logger.info(f"🧠 XGBoost trained - Score: {final_score:.3f}")
        
        return final_model, final_score
    
    def _train_advanced_catboost(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
        """Train advanced CatBoost with sophisticated hyperparameter optimization"""
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'random_state': 42,
                'verbose': False,
                'eval_metric': 'RMSE'
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model = cb.CatBoostRegressor(**params)
                if len(X_val) > 0:
                    model.fit(X_train, y_train,
                              eval_set=[(X_val, y_val)],
                              early_stopping_rounds=50,
                              use_best_model=True,
                              eval_metric='RMSE',
                              verbose=False)
                else:
                    model.fit(X_train, y_train, verbose=False)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'random_state': 42,
            'verbose': False,
            'eval_metric': 'RMSE'
        })
        
        # Filter best_params to only include valid CatBoostRegressor params
        valid_keys = set(cb.CatBoostRegressor().get_params().keys())
        best_params = {k: v for k, v in best_params.items() if k in valid_keys}

        # Remove early stopping and eval_set for final fit
        for k in ['early_stopping_rounds', 'use_best_model', 'eval_set']:
            best_params.pop(k, None)

        # Final model: create new CatBoostRegressor and fit with no eval_set or early stopping
        final_model = cb.CatBoostRegressor(**best_params)
        final_model.fit(X, y)
        
        # Calculate final score
        y_pred = final_model.predict(X)
        final_score = r2_score(y, y_pred)
        
        self.hyperparameters['catboost'] = best_params
        logger.info(f"🧠 CatBoost trained - Score: {final_score:.3f}")
        
        return final_model, final_score
    
    def _train_advanced_neural_network(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
        """Train advanced neural network with sophisticated architecture"""
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        def create_advanced_model():
            model = Sequential([
                # Input layer
                Dense(512, activation='relu', input_shape=(X.shape[1],), 
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.3),
                
                # Hidden layers
                Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.1),
                
                # Output layer
                Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            return model
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = create_advanced_model()
            
            # Advanced callbacks
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7),
                ModelCheckpoint('best_neural_model.h5', save_best_only=True)
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            y_pred = model.predict(X_val, verbose=0)
            score = r2_score(y_val, y_pred.flatten())
            scores.append(score)
        
        # Train final model
        final_model = create_advanced_model()
        final_model.fit(X_scaled, y, epochs=200, batch_size=32, verbose=0)
        
        # Calculate final score
        y_pred = final_model.predict(X_scaled, verbose=0)
        final_score = r2_score(y, y_pred.flatten())
        
        # Store scaler with model
        final_model.scaler = scaler
        
        logger.info(f"🧠 Neural Network trained - Score: {final_score:.3f}")
        
        return final_model, final_score
    
    def _train_advanced_lstm(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
        """Train advanced LSTM for time series prediction"""
        
        # Reshape data for LSTM (samples, timesteps, features)
        def create_sequences(X, y, time_steps=10):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X.iloc[i:(i + time_steps)].values)
                ys.append(y.iloc[i + time_steps])
            return np.array(Xs), np.array(ys)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        time_steps = 10
        X_seq, y_seq = create_sequences(X, y, time_steps)
        
        def create_lstm_model():
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(time_steps, X.shape[1])),
                Dropout(0.3),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_seq):
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = y_seq[train_idx], y_seq[val_idx]
            
            model = create_lstm_model()
            
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7)
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=150,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            y_pred = model.predict(X_val, verbose=0)
            score = r2_score(y_val, y_pred.flatten())
            scores.append(score)
        
        # Train final model
        final_model = create_lstm_model()
        final_model.fit(X_seq, y_seq, epochs=150, batch_size=32, verbose=0)
        
        # Calculate final score
        y_pred = final_model.predict(X_seq, verbose=0)
        final_score = r2_score(y_seq, y_pred.flatten())
        
        # Store scaler and time_steps with model
        final_model.scaler = scaler
        final_model.time_steps = time_steps
        
        logger.info(f"🧠 LSTM trained - Score: {final_score:.3f}")
        
        return final_model, final_score
    
    def _train_advanced_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
        """Train advanced Random Forest with sophisticated hyperparameter optimization"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
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
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params['random_state'] = 42
        
        final_model = RandomForestRegressor(**best_params)
        final_model.fit(X, y)
        
        # Calculate final score
        y_pred = final_model.predict(X)
        final_score = r2_score(y, y_pred)
        
        self.hyperparameters['random_forest'] = best_params
        logger.info(f"🧠 Random Forest trained - Score: {final_score:.3f}")
        
        return final_model, final_score
    
    def _train_advanced_gradient_boosting(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
        """Train advanced Gradient Boosting with sophisticated hyperparameter optimization"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': 42
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = GradientBoostingRegressor(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params['random_state'] = 42
        
        final_model = GradientBoostingRegressor(**best_params)
        final_model.fit(X, y)
        
        # Calculate final score
        y_pred = final_model.predict(X)
        final_score = r2_score(y, y_pred)
        
        self.hyperparameters['gradient_boosting'] = best_params
        logger.info(f"🧠 Gradient Boosting trained - Score: {final_score:.3f}")
        
        return final_model, final_score
    
    def _create_intelligent_ensemble(self) -> Dict[str, float]:
        """Create intelligent ensemble weights based on performance and diversity"""
        
        # Calculate base weights from performance
        performance_weights = {}
        total_performance = sum(self.model_performance.values())
        
        for model_name, score in self.model_performance.items():
            performance_weights[model_name] = score / total_performance
        
        # Apply diversity bonus (favor different model types)
        diversity_multipliers = {
            'lightgbm': 1.1,  # Tree-based
            'xgboost': 1.1,   # Tree-based
            'catboost': 1.1,  # Tree-based
            'neural_network': 1.2,  # Neural
            'lstm': 1.2,      # Neural
            'random_forest': 1.0,   # Ensemble
            'gradient_boosting': 1.0  # Ensemble
        }
        
        # Calculate final weights
        final_weights = {}
        for model_name in self.model_performance.keys():
            base_weight = performance_weights[model_name]
            diversity_bonus = diversity_multipliers.get(model_name, 1.0)
            final_weights[model_name] = base_weight * diversity_bonus
        
        # Normalize weights
        total_weight = sum(final_weights.values())
        final_weights = {k: v / total_weight for k, v in final_weights.items()}
        
        self.ensemble_weights = final_weights
        
        logger.info("🧠 Intelligent ensemble weights calculated:")
        for model_name, weight in final_weights.items():
            logger.info(f"   • {model_name}: {weight:.3f}")
        
        return final_weights
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble prediction using intelligent weights"""
        
        predictions = {}
        
        for model_name, model in self.models.items():
            if model_name in ['neural_network', 'lstm']:
                # Handle neural networks with scaling
                if hasattr(model, 'scaler'):
                    X_scaled = model.scaler.transform(X)
                    if model_name == 'lstm':
                        # Handle LSTM sequence creation
                        X_seq = self._create_sequences_for_lstm(X_scaled, model.time_steps)
                        pred = model.predict(X_seq, verbose=0).flatten()
                        # Pad with zeros for missing predictions
                        full_pred = np.zeros(len(X))
                        full_pred[model.time_steps:] = pred
                        predictions[model_name] = full_pred
                    else:
                        predictions[model_name] = model.predict(X_scaled, verbose=0).flatten()
                else:
                    predictions[model_name] = model.predict(X, verbose=0).flatten()
            else:
                predictions[model_name] = model.predict(X)
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for model_name, pred in predictions.items():
            weight = self.ensemble_weights.get(model_name, 0)
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def _create_sequences_for_lstm(self, X: np.ndarray, time_steps: int) -> np.ndarray:
        """Create sequences for LSTM prediction"""
        Xs = []
        for i in range(len(X) - time_steps + 1):
            Xs.append(X[i:(i + time_steps)])
        return np.array(Xs)
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model performance summary"""
        return {
            'model_performance': self.model_performance,
            'ensemble_weights': self.ensemble_weights,
            'hyperparameters': self.hyperparameters,
            'best_model': max(self.model_performance, key=self.model_performance.get),
            'average_score': np.mean(list(self.model_performance.values())),
            'score_std': np.std(list(self.model_performance.values()))
        } 