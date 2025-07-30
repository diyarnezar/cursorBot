"""
Advanced Ensemble Techniques for Maximum Prediction Accuracy
Part of Project Hyperion - Ultimate Autonomous Trading Bot

Implements:
- Advanced Stacking with multiple meta-learners
- Blending with time-based validation
- Bagging with diversity optimization
- Boosting with adaptive weights
- Voting with confidence weighting
- Dynamic ensemble selection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, BaggingRegressor, AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)


class AdvancedEnsemble:
    """
    Advanced Ensemble Methods for Maximum Prediction Accuracy
    
    Features:
    - Multi-level stacking with diverse base models
    - Time-aware blending with proper validation
    - Bagging with diversity optimization
    - Boosting with adaptive learning rates
    - Voting with confidence-based weighting
    - Dynamic ensemble selection
    - Ensemble diversity metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_models = {}
        self.meta_models = {}
        self.ensemble_weights = {}
        self.performance_metrics = {}
        self.ensemble_history = []
        
        # Ensemble parameters
        self.n_base_models = config.get('n_base_models', 10)
        self.n_meta_models = config.get('n_meta_models', 5)
        self.cv_folds = config.get('cv_folds', 5)
        self.diversity_threshold = config.get('diversity_threshold', 0.3)
        
        logger.info("Advanced Ensemble initialized")

    def create_base_models(self) -> Dict[str, Any]:
        """Create diverse base models for ensemble"""
        try:
            base_models = {}
            
            # Tree-based models
            base_models['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            base_models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            base_models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            base_models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            base_models['catboost'] = CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            )
            
            # Linear models
            base_models['linear_regression'] = LinearRegression()
            base_models['ridge'] = Ridge(alpha=1.0)
            base_models['lasso'] = Lasso(alpha=0.1)
            base_models['elastic_net'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
            
            # Support Vector Regression
            base_models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale')
            
            # Neural Network
            base_models['mlp'] = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
            
            # AdaBoost
            base_models['adaboost'] = AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
            
            self.base_models = base_models
            logger.info(f"Created {len(base_models)} base models")
            return base_models
            
        except Exception as e:
            logger.error(f"Error creating base models: {e}")
            return {}

    def create_meta_models(self) -> Dict[str, Any]:
        """Create meta-learners for stacking"""
        try:
            meta_models = {}
            
            # Linear meta-learners
            meta_models['linear'] = LinearRegression()
            meta_models['ridge'] = Ridge(alpha=1.0)
            meta_models['lasso'] = Lasso(alpha=0.1)
            meta_models['elastic_net'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
            
            # Non-linear meta-learners
            meta_models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale')
            meta_models['mlp'] = MLPRegressor(
                hidden_layer_sizes=(50, 25),
                activation='relu',
                solver='adam',
                max_iter=300,
                random_state=42
            )
            
            # Tree-based meta-learners
            meta_models['random_forest_meta'] = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
            
            meta_models['lightgbm_meta'] = lgb.LGBMRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                verbose=-1
            )
            
            self.meta_models = meta_models
            logger.info(f"Created {len(meta_models)} meta models")
            return meta_models
            
        except Exception as e:
            logger.error(f"Error creating meta models: {e}")
            return {}

    def advanced_stacking(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Advanced stacking with multiple meta-learners"""
        try:
            logger.info("Starting advanced stacking ensemble")
            
            # Train base models
            base_predictions_train = {}
            base_predictions_val = {}
            
            for name, model in self.base_models.items():
                logger.info(f"Training base model: {name}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Generate predictions
                base_predictions_train[name] = model.predict(X_train)
                base_predictions_val[name] = model.predict(X_val)
            
            # Create meta-features
            meta_features_train = np.column_stack(list(base_predictions_train.values()))
            meta_features_val = np.column_stack(list(base_predictions_val.values()))
            
            # Train meta-models
            meta_predictions = {}
            meta_scores = {}
            
            for name, meta_model in self.meta_models.items():
                logger.info(f"Training meta model: {name}")
                
                # Train meta-model
                meta_model.fit(meta_features_train, y_train)
                
                # Generate meta-predictions
                meta_predictions[name] = meta_model.predict(meta_features_val)
                
                # Calculate meta-model score
                meta_scores[name] = r2_score(y_val, meta_predictions[name])
            
            # Select best meta-model
            best_meta_name = max(meta_scores, key=meta_scores.get)
            best_meta_model = self.meta_models[best_meta_name]
            
            # Final ensemble prediction
            final_prediction = meta_predictions[best_meta_name]
            
            # Calculate ensemble metrics
            ensemble_score = r2_score(y_val, final_prediction)
            ensemble_mse = mean_squared_error(y_val, final_prediction)
            ensemble_mae = mean_absolute_error(y_val, final_prediction)
            
            result = {
                'ensemble_type': 'advanced_stacking',
                'best_meta_model': best_meta_name,
                'ensemble_score': ensemble_score,
                'ensemble_mse': ensemble_mse,
                'ensemble_mae': ensemble_mae,
                'meta_scores': meta_scores,
                'base_models': list(self.base_models.keys()),
                'meta_models': list(self.meta_models.keys()),
                'final_prediction': final_prediction
            }
            
            logger.info(f"Advanced stacking completed. Score: {ensemble_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced stacking: {e}")
            return {}

    def time_aware_blending(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Time-aware blending with proper validation"""
        try:
            logger.info("Starting time-aware blending ensemble")
            
            # Train base models
            base_predictions_train = {}
            base_predictions_val = {}
            base_scores = {}
            
            for name, model in self.base_models.items():
                logger.info(f"Training base model for blending: {name}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Generate predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                base_predictions_train[name] = train_pred
                base_predictions_val[name] = val_pred
                
                # Calculate base model score
                base_scores[name] = r2_score(y_val, val_pred)
            
            # Calculate blending weights based on validation performance
            total_score = sum(max(score, 0) for score in base_scores.values())
            
            if total_score > 0:
                blending_weights = {
                    name: max(score, 0) / total_score 
                    for name, score in base_scores.items()
                }
            else:
                # Equal weights if all scores are negative
                blending_weights = {
                    name: 1.0 / len(base_scores) 
                    for name in base_scores.keys()
                }
            
            # Generate blended prediction
            blended_prediction = np.zeros_like(y_val)
            for name, weight in blending_weights.items():
                blended_prediction += weight * base_predictions_val[name]
            
            # Calculate ensemble metrics
            ensemble_score = r2_score(y_val, blended_prediction)
            ensemble_mse = mean_squared_error(y_val, blended_prediction)
            ensemble_mae = mean_absolute_error(y_val, blended_prediction)
            
            result = {
                'ensemble_type': 'time_aware_blending',
                'blending_weights': blending_weights,
                'base_scores': base_scores,
                'ensemble_score': ensemble_score,
                'ensemble_mse': ensemble_mse,
                'ensemble_mae': ensemble_mae,
                'base_models': list(self.base_models.keys()),
                'final_prediction': blended_prediction
            }
            
            logger.info(f"Time-aware blending completed. Score: {ensemble_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in time-aware blending: {e}")
            return {}

    def diversity_optimized_bagging(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Bagging with diversity optimization"""
        try:
            logger.info("Starting diversity-optimized bagging")
            
            # Create bagging models with different base estimators
            bagging_models = {}
            
            # Random Forest Bagging
            bagging_models['rf_bagging'] = BaggingRegressor(
                base_estimator=RandomForestRegressor(n_estimators=50, random_state=42),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=42
            )
            
            # Linear Regression Bagging
            bagging_models['linear_bagging'] = BaggingRegressor(
                base_estimator=LinearRegression(),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=42
            )
            
            # SVR Bagging
            bagging_models['svr_bagging'] = BaggingRegressor(
                base_estimator=SVR(kernel='rbf', C=1.0),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=42
            )
            
            # Train bagging models
            bagging_predictions = {}
            bagging_scores = {}
            
            for name, model in bagging_models.items():
                logger.info(f"Training bagging model: {name}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Generate predictions
                predictions = model.predict(X_val)
                bagging_predictions[name] = predictions
                
                # Calculate score
                bagging_scores[name] = r2_score(y_val, predictions)
            
            # Calculate diversity between models
            diversity_matrix = self._calculate_diversity_matrix(bagging_predictions)
            
            # Select diverse models
            diverse_models = self._select_diverse_models(bagging_predictions, diversity_matrix)
            
            # Generate ensemble prediction
            ensemble_prediction = np.mean([bagging_predictions[name] for name in diverse_models], axis=0)
            
            # Calculate ensemble metrics
            ensemble_score = r2_score(y_val, ensemble_prediction)
            ensemble_mse = mean_squared_error(y_val, ensemble_prediction)
            ensemble_mae = mean_absolute_error(y_val, ensemble_prediction)
            
            result = {
                'ensemble_type': 'diversity_optimized_bagging',
                'bagging_scores': bagging_scores,
                'diverse_models': diverse_models,
                'diversity_matrix': diversity_matrix,
                'ensemble_score': ensemble_score,
                'ensemble_mse': ensemble_mse,
                'ensemble_mae': ensemble_mae,
                'final_prediction': ensemble_prediction
            }
            
            logger.info(f"Diversity-optimized bagging completed. Score: {ensemble_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in diversity-optimized bagging: {e}")
            return {}

    def adaptive_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Adaptive boosting with dynamic weight adjustment"""
        try:
            logger.info("Starting adaptive boosting ensemble")
            
            # Create boosting models
            boosting_models = {}
            
            # AdaBoost with different base estimators
            boosting_models['adaboost_rf'] = AdaBoostRegressor(
                base_estimator=RandomForestRegressor(n_estimators=50, random_state=42),
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
            
            boosting_models['adaboost_svr'] = AdaBoostRegressor(
                base_estimator=SVR(kernel='rbf', C=1.0),
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
            
            # Gradient Boosting variants
            boosting_models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            boosting_models['lightgbm_boost'] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            # Train boosting models
            boosting_predictions = {}
            boosting_scores = {}
            
            for name, model in boosting_models.items():
                logger.info(f"Training boosting model: {name}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Generate predictions
                predictions = model.predict(X_val)
                boosting_predictions[name] = predictions
                
                # Calculate score
                boosting_scores[name] = r2_score(y_val, predictions)
            
            # Calculate adaptive weights based on performance
            total_score = sum(max(score, 0) for score in boosting_scores.values())
            
            if total_score > 0:
                adaptive_weights = {
                    name: max(score, 0) / total_score 
                    for name, score in boosting_scores.items()
                }
            else:
                adaptive_weights = {
                    name: 1.0 / len(boosting_scores) 
                    for name in boosting_scores.keys()
                }
            
            # Generate ensemble prediction
            ensemble_prediction = np.zeros_like(y_val)
            for name, weight in adaptive_weights.items():
                ensemble_prediction += weight * boosting_predictions[name]
            
            # Calculate ensemble metrics
            ensemble_score = r2_score(y_val, ensemble_prediction)
            ensemble_mse = mean_squared_error(y_val, ensemble_prediction)
            ensemble_mae = mean_absolute_error(y_val, ensemble_prediction)
            
            result = {
                'ensemble_type': 'adaptive_boosting',
                'adaptive_weights': adaptive_weights,
                'boosting_scores': boosting_scores,
                'ensemble_score': ensemble_score,
                'ensemble_mse': ensemble_mse,
                'ensemble_mae': ensemble_mae,
                'final_prediction': ensemble_prediction
            }
            
            logger.info(f"Adaptive boosting completed. Score: {ensemble_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in adaptive boosting: {e}")
            return {}

    def confidence_weighted_voting(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Voting ensemble with confidence-based weighting"""
        try:
            logger.info("Starting confidence-weighted voting")
            
            # Train base models
            base_predictions = {}
            base_scores = {}
            base_confidences = {}
            
            for name, model in self.base_models.items():
                logger.info(f"Training base model for voting: {name}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Generate predictions
                predictions = model.predict(X_val)
                base_predictions[name] = predictions
                
                # Calculate score
                score = r2_score(y_val, predictions)
                base_scores[name] = score
                
                # Calculate confidence (based on prediction stability)
                confidence = self._calculate_prediction_confidence(predictions, y_val)
                base_confidences[name] = confidence
            
            # Calculate voting weights based on confidence
            total_confidence = sum(base_confidences.values())
            
            if total_confidence > 0:
                voting_weights = {
                    name: confidence / total_confidence 
                    for name, confidence in base_confidences.items()
                }
            else:
                voting_weights = {
                    name: 1.0 / len(base_confidences) 
                    for name in base_confidences.keys()
                }
            
            # Generate weighted voting prediction
            ensemble_prediction = np.zeros_like(y_val)
            for name, weight in voting_weights.items():
                ensemble_prediction += weight * base_predictions[name]
            
            # Calculate ensemble metrics
            ensemble_score = r2_score(y_val, ensemble_prediction)
            ensemble_mse = mean_squared_error(y_val, ensemble_prediction)
            ensemble_mae = mean_absolute_error(y_val, ensemble_prediction)
            
            result = {
                'ensemble_type': 'confidence_weighted_voting',
                'voting_weights': voting_weights,
                'base_scores': base_scores,
                'base_confidences': base_confidences,
                'ensemble_score': ensemble_score,
                'ensemble_mse': ensemble_mse,
                'ensemble_mae': ensemble_mae,
                'final_prediction': ensemble_prediction
            }
            
            logger.info(f"Confidence-weighted voting completed. Score: {ensemble_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in confidence-weighted voting: {e}")
            return {}

    def _calculate_diversity_matrix(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate diversity matrix between model predictions"""
        try:
            n_models = len(predictions)
            diversity_matrix = np.zeros((n_models, n_models))
            
            model_names = list(predictions.keys())
            
            for i, name1 in enumerate(model_names):
                for j, name2 in enumerate(model_names):
                    if i != j:
                        # Calculate correlation between predictions
                        correlation = np.corrcoef(predictions[name1], predictions[name2])[0, 1]
                        # Convert to diversity (1 - abs(correlation))
                        diversity_matrix[i, j] = 1 - abs(correlation)
                    else:
                        diversity_matrix[i, j] = 0
            
            return diversity_matrix
            
        except Exception as e:
            logger.error(f"Error calculating diversity matrix: {e}")
            return np.zeros((len(predictions), len(predictions)))

    def _select_diverse_models(self, predictions: Dict[str, np.ndarray], 
                              diversity_matrix: np.ndarray) -> List[str]:
        """Select diverse models based on diversity matrix"""
        try:
            model_names = list(predictions.keys())
            n_models = len(model_names)
            
            # Start with the first model
            selected_models = [model_names[0]]
            
            # Add models that are diverse from already selected ones
            for i in range(1, n_models):
                model_name = model_names[i]
                
                # Calculate average diversity with selected models
                avg_diversity = 0
                for selected_model in selected_models:
                    selected_idx = model_names.index(selected_model)
                    current_idx = model_names.index(model_name)
                    avg_diversity += diversity_matrix[selected_idx, current_idx]
                
                avg_diversity /= len(selected_models)
                
                # Add model if it's diverse enough
                if avg_diversity >= self.diversity_threshold:
                    selected_models.append(model_name)
            
            return selected_models
            
        except Exception as e:
            logger.error(f"Error selecting diverse models: {e}")
            return list(predictions.keys())

    def _calculate_prediction_confidence(self, predictions: np.ndarray, 
                                       actual: np.ndarray) -> float:
        """Calculate prediction confidence based on stability and accuracy"""
        try:
            # Calculate prediction stability (variance)
            prediction_variance = np.var(predictions)
            
            # Calculate prediction accuracy
            prediction_accuracy = r2_score(actual, predictions)
            
            # Combine stability and accuracy for confidence
            confidence = (1 - prediction_variance / np.var(actual)) * max(prediction_accuracy, 0)
            
            return max(confidence, 0.1)  # Minimum confidence threshold
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
            return 0.5

    def run_all_ensembles(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Run all ensemble methods and compare performance"""
        try:
            logger.info("Running all ensemble methods")
            
            # Create models
            self.create_base_models()
            self.create_meta_models()
            
            # Run all ensemble methods
            results = {}
            
            results['stacking'] = self.advanced_stacking(X_train, y_train, X_val, y_val)
            results['blending'] = self.time_aware_blending(X_train, y_train, X_val, y_val)
            results['bagging'] = self.diversity_optimized_bagging(X_train, y_train, X_val, y_val)
            results['boosting'] = self.adaptive_boosting(X_train, y_train, X_val, y_val)
            results['voting'] = self.confidence_weighted_voting(X_train, y_train, X_val, y_val)
            
            # Find best ensemble
            ensemble_scores = {
                name: result.get('ensemble_score', -1) 
                for name, result in results.items()
            }
            
            best_ensemble = max(ensemble_scores, key=ensemble_scores.get)
            best_score = ensemble_scores[best_ensemble]
            
            # Store results
            self.ensemble_history.append({
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'best_ensemble': best_ensemble,
                'best_score': best_score
            })
            
            final_result = {
                'all_results': results,
                'best_ensemble': best_ensemble,
                'best_score': best_score,
                'ensemble_comparison': ensemble_scores
            }
            
            logger.info(f"All ensembles completed. Best: {best_ensemble} (Score: {best_score:.4f})")
            return final_result
            
        except Exception as e:
            logger.error(f"Error running all ensembles: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    config = {
        'n_base_models': 10,
        'n_meta_models': 5,
        'cv_folds': 5,
        'diversity_threshold': 0.3
    }
    
    ensemble = AdvancedEnsemble(config)
    
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randn(1000)
    X_val = np.random.randn(200, 20)
    y_val = np.random.randn(200)
    
    # Run all ensemble methods
    results = ensemble.run_all_ensembles(X_train, y_train, X_val, y_val) 