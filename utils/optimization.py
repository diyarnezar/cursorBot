"""
Hyperparameter Optimizer for Project Hyperion
Advanced hyperparameter optimization and model tuning
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import joblib


class HyperparameterOptimizer:
    """
    Advanced Hyperparameter Optimizer for model optimization
    Features: Optuna optimization, Bayesian optimization, automated tuning
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize hyperparameter optimizer"""
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.config = config or {}
        
        # Optimization parameters
        self.max_trials = 100
        self.timeout = 3600  # 1 hour
        self.n_jobs = -1
        
        # Optimization history
        self.optimization_history = {}
        self.best_params = {}
        
        self.logger.info("🔧 Hyperparameter Optimizer initialized")
    
    def optimize_models(self, models: Dict[str, Any], features: pd.DataFrame, target: pd.Series, max_iterations: int = 50) -> Dict[str, Any]:
        """Optimize all models using advanced techniques"""
        self.logger.info(f"🔧 Starting hyperparameter optimization with {max_iterations} iterations")
        
        try:
            optimized_models = models.copy()
            
            # Filter out non-numeric columns to prevent timestamp errors
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            non_numeric_columns = features.select_dtypes(exclude=[np.number]).columns
            
            if len(non_numeric_columns) > 0:
                self.logger.info(f"Removing {len(non_numeric_columns)} non-numeric columns for hyperparameter optimization: {list(non_numeric_columns)}")
                features = features[numeric_columns]
            
            # Ensure target is properly formatted and aligned with features
            if isinstance(target, pd.Series):
                # Align target with features index
                if not target.index.equals(features.index):
                    self.logger.info("Aligning target with features index for hyperparameter optimization")
                    common_index = features.index.intersection(target.index)
                    features_aligned = features.loc[common_index]
                    target_aligned = target.loc[common_index]
                else:
                    features_aligned = features
                    target_aligned = target
                target_values = target_aligned.values
            else:
                target_values = target
                features_aligned = features
            
            # Create features with target for optimization
            features_with_target = features_aligned.copy()
            features_with_target['target'] = target_values
            
            # Ensure no NaN values
            features_with_target = features_with_target.dropna()
            if len(features_with_target) == 0:
                self.logger.warning("No valid data after alignment for hyperparameter optimization")
                return optimized_models
            
            self.logger.info(f"Hyperparameter optimization data shape: {features_with_target.shape}")
            
            # Optimize each model
            for model_name, model in optimized_models.items():
                try:
                    self.logger.info(f"🔧 Optimizing {model_name}...")
                    
                    # Create study for this model
                    study = optuna.create_study(direction='minimize')
                    
                    # Define objective function
                    def objective(trial):
                        # Get hyperparameters for this trial
                        params = self._get_hyperparameters(trial, model_name)
                        
                        # Create model with new parameters
                        optimized_model = self._create_model_with_params(model_name, params)
                        
                        # Evaluate model
                        score = self._evaluate_model(optimized_model, features_with_target)
                        
                        return score
                    
                    # Run optimization
                    study.optimize(objective, n_trials=max_iterations, show_progress_bar=False)
                    
                    # Get best parameters and update model
                    best_params = study.best_params
                    optimized_models[model_name] = self._create_model_with_params(model_name, best_params)
                    
                    self.logger.info(f"✅ {model_name} optimized with score: {study.best_value:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to optimize {model_name}: {e}")
                    continue
            
            self.logger.info("✅ Hyperparameter optimization completed successfully")
            return optimized_models
            
        except Exception as e:
            self.logger.error(f"❌ Hyperparameter optimization failed: {e}")
            return models
    
    def _objective_function(self, trial, model, features: pd.DataFrame, model_name: str) -> float:
        """Objective function for optimization"""
        try:
            # Get hyperparameter suggestions based on model type
            params = self._suggest_hyperparameters(trial, model_name)
            
            # Create model with suggested parameters
            optimized_model = self._create_model_with_params(model_name, params)
            
            # Prepare data
            X = features.drop('target', axis=1, errors='ignore')
            y = features['target']
            
            # Split data for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            optimized_model.fit(X_train, y_train)
            
            # Predict and calculate score
            y_pred = optimized_model.predict(X_val)
            score = r2_score(y_val, y_pred)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"🔧 Objective function failed for {model_name}: {str(e)}")
            return -1.0
    
    def _suggest_hyperparameters(self, trial, model_name: str) -> Dict[str, Any]:
        """Suggest hyperparameters based on model type"""
        params = {}
        
        if 'random_forest' in model_name.lower() or 'rf' in model_name.lower():
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 30)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
            params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            
        elif 'gradient_boosting' in model_name.lower() or 'gb' in model_name.lower():
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            
        elif 'neural_network' in model_name.lower() or 'mlp' in model_name.lower():
            # Suggest network architecture
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_sizes = []
            for i in range(n_layers):
                hidden_sizes.append(trial.suggest_int(f'hidden_size_{i}', 10, 200))
            
            params['hidden_layer_sizes'] = tuple(hidden_sizes)
            params['learning_rate_init'] = trial.suggest_float('learning_rate_init', 0.0001, 0.1, log=True)
            params['alpha'] = trial.suggest_float('alpha', 0.0001, 0.1, log=True)
            params['max_iter'] = trial.suggest_int('max_iter', 200, 1000)
            
        elif 'lstm' in model_name.lower() or 'gru' in model_name.lower():
            params['units'] = trial.suggest_int('units', 32, 256)
            params['layers'] = trial.suggest_int('layers', 1, 4)
            params['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
            
        else:
            # Default parameters for unknown model types
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
        
        return params
    
    def _get_hyperparameters(self, trial, model_name: str) -> Dict[str, Any]:
        """Get hyperparameters for a specific model type"""
        try:
            model_name_lower = model_name.lower()
            
            if 'lightgbm' in model_name_lower or 'lgb' in model_name_lower:
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                }
            elif 'xgboost' in model_name_lower or 'xgb' in model_name_lower:
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                }
            elif 'random_forest' in model_name_lower or 'rf' in model_name_lower or 'forest' in model_name_lower:
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
            elif 'gradient' in model_name_lower or 'gbm' in model_name_lower:
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            else:
                # Default parameters for unknown models (no learning_rate to be safe)
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10)
                }
        except Exception as e:
            self.logger.warning(f"Hyperparameter generation failed for {model_name}: {e}")
            return {}
    
    def _create_model_with_params(self, model_name: str, params: Dict[str, Any]):
        """Create a model with the given parameters"""
        try:
            model_name_lower = model_name.lower()
            
            if 'lightgbm' in model_name_lower or 'lgb' in model_name_lower:
                import lightgbm as lgb
                return lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
            elif 'xgboost' in model_name_lower or 'xgb' in model_name_lower:
                import xgboost as xgb
                return xgb.XGBRegressor(**params, random_state=42)
            elif 'random_forest' in model_name_lower or 'rf' in model_name_lower or 'forest' in model_name_lower:
                from sklearn.ensemble import RandomForestRegressor
                # Ensure no learning_rate for Random Forest
                params_clean = {k: v for k, v in params.items() if k != 'learning_rate'}
                return RandomForestRegressor(**params_clean, random_state=42)
            elif 'gradient' in model_name_lower or 'gbm' in model_name_lower:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(**params, random_state=42)
            else:
                # Default to Random Forest (safe choice)
                from sklearn.ensemble import RandomForestRegressor
                # Ensure no learning_rate for Random Forest
                params_clean = {k: v for k, v in params.items() if k != 'learning_rate'}
                return RandomForestRegressor(**params_clean, random_state=42)
        except Exception as e:
            self.logger.warning(f"Model creation failed for {model_name}: {e}")
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(random_state=42)
    
    def _update_model_with_params(self, model, params: Dict[str, Any]):
        """Update existing model with new parameters"""
        try:
            # Try to set parameters directly
            model.set_params(**params)
            return model
        except:
            # If that fails, create new model instance
            return self._create_model_with_params(type(model).__name__, params)
    
    def optimize_single_model(self, model, features: pd.DataFrame, model_name: str = "model") -> Any:
        """Optimize a single model"""
        self.logger.info(f"🔧 Optimizing single model: {model_name}")
        
        # Create optimization study
        study = optuna.create_study(direction='maximize')
        
        # Define objective function
        def objective(trial):
            return self._objective_function(trial, model, features, model_name)
        
        # Run optimization
        study.optimize(objective, n_trials=self.max_trials, timeout=self.timeout)
        
        # Get best parameters and update model
        best_params = study.best_params
        optimized_model = self._update_model_with_params(model, best_params)
        
        self.logger.info(f"🔧 Single model optimization completed. Best score: {study.best_value:.4f}")
        return optimized_model
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        summary = {
            'total_models_optimized': len(self.optimization_history),
            'best_params': self.best_params,
            'optimization_history': self.optimization_history,
            'average_improvement': 0.0
        }
        
        if self.optimization_history:
            scores = [history['best_score'] for history in self.optimization_history.values()]
            summary['average_improvement'] = np.mean(scores)
            summary['best_score'] = max(scores)
            summary['worst_score'] = min(scores)
        
        return summary
    
    def save_optimization_state(self, filepath: str):
        """Save optimization state"""
        state_data = {
            'optimization_history': self.optimization_history,
            'best_params': self.best_params,
            'config': self.config
        }
        joblib.dump(state_data, filepath)
        self.logger.info(f"🔧 Optimization state saved to {filepath}")
    
    def load_optimization_state(self, filepath: str):
        """Load optimization state"""
        state_data = joblib.load(filepath)
        self.optimization_history = state_data['optimization_history']
        self.best_params = state_data['best_params']
        self.config = state_data['config']
        self.logger.info(f"🔧 Optimization state loaded from {filepath}")
    
    def suggest_optimization_strategy(self) -> List[str]:
        """Suggest optimization strategies based on current state"""
        suggestions = []
        
        if not self.optimization_history:
            suggestions.append("Start with basic hyperparameter optimization")
            suggestions.append("Use Optuna for automated parameter tuning")
            suggestions.append("Consider ensemble methods for better performance")
        else:
            avg_score = np.mean([h['best_score'] for h in self.optimization_history.values()])
            
            if avg_score < 0.5:
                suggestions.append("Consider feature engineering improvements")
                suggestions.append("Try different model architectures")
                suggestions.append("Increase optimization trials")
            elif avg_score > 0.8:
                suggestions.append("Fine-tune with smaller parameter ranges")
                suggestions.append("Consider ensemble optimization")
                suggestions.append("Explore advanced optimization techniques")
        
        return suggestions 

    def _evaluate_model(self, model, features_with_target: pd.DataFrame) -> float:
        """Evaluate model performance and return score"""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import mean_squared_error
            
            # Prepare data
            X = features_with_target.drop('target', axis=1, errors='ignore')
            y = features_with_target['target']
            
            # Filter out non-numeric columns to prevent timestamp errors
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
            
            if len(non_numeric_columns) > 0:
                self.logger.info(f"Removing {len(non_numeric_columns)} non-numeric columns for hyperparameter optimization: {list(non_numeric_columns)}")
                X = X[numeric_columns]
            
            # Ensure no NaN values
            X = X.dropna()
            # Use position-based indexing instead of label-based indexing
            y = y.iloc[:len(X)]
            
            if len(X) == 0:
                self.logger.warning("No valid data for hyperparameter optimization evaluation")
                return 1.0
            
            # Use cross-validation for evaluation
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            
            # Convert to positive score (lower is better)
            mse = -np.mean(scores)
            return mse
            
        except Exception as e:
            self.logger.warning(f"Model evaluation failed: {e}")
            return 1.0  # High error score 