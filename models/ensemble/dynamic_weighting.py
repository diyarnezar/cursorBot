"""
Dynamic Weighting System for Real-time Ensemble Optimization
Part of Project Hyperion - Ultimate Autonomous Trading Bot

Features:
- Real-time weight optimization
- Performance-based adaptation
- Market condition awareness
- Confidence-based weighting
- Adaptive learning rates
- Portfolio optimization integration
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)


class DynamicWeighting:
    """
    Dynamic Weighting System for Real-time Ensemble Optimization
    
    Features:
    - Real-time weight optimization based on recent performance
    - Market condition-aware weighting
    - Confidence-based weight adjustment
    - Adaptive learning rates
    - Portfolio optimization integration
    - Multi-objective optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights_history = []
        self.performance_history = []
        self.market_conditions = {}
        self.optimization_results = {}
        
        # Dynamic weighting parameters
        self.learning_rate = config.get('learning_rate', 0.01)
        self.adaptation_window = config.get('adaptation_window', 100)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.optimization_method = config.get('optimization_method', 'sharpe')
        
        # Weight constraints
        self.min_weight = config.get('min_weight', 0.01)
        self.max_weight = config.get('max_weight', 0.5)
        
        logger.info("Dynamic Weighting System initialized")

    def optimize_weights_performance(self, model_predictions: Dict[str, np.ndarray],
                                   actual_values: np.ndarray,
                                   current_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Optimize ensemble weights based on recent performance"""
        try:
            logger.info("Optimizing weights based on performance")
            
            # Initialize weights if not provided
            if current_weights is None:
                n_models = len(model_predictions)
                current_weights = {name: 1.0 / n_models for name in model_predictions.keys()}
            
            # Calculate individual model performance
            model_performance = {}
            for name, predictions in model_predictions.items():
                mse = mean_squared_error(actual_values, predictions)
                r2 = r2_score(actual_values, predictions)
                model_performance[name] = {
                    'mse': mse,
                    'r2': r2,
                    'performance_score': 1.0 / (1.0 + mse)  # Higher score for lower MSE
                }
            
            # Define optimization objective
            def objective_function(weights_array):
                weights_dict = dict(zip(model_predictions.keys(), weights_array))
                
                # Calculate ensemble prediction
                ensemble_prediction = np.zeros_like(actual_values)
                for name, weight in weights_dict.items():
                    ensemble_prediction += weight * model_predictions[name]
                
                # Calculate ensemble performance
                ensemble_mse = mean_squared_error(actual_values, ensemble_prediction)
                ensemble_r2 = r2_score(actual_values, ensemble_prediction)
                
                # Multi-objective: minimize MSE and maximize R²
                objective = ensemble_mse - 0.1 * ensemble_r2
                
                return objective
            
            # Define constraints
            n_models = len(model_predictions)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Define bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_models)]
            
            # Initial weights
            initial_weights = np.array(list(current_weights.values()))
            
            # Optimize weights
            result = minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimized_weights = dict(zip(model_predictions.keys(), result.x))
                
                # Calculate optimized ensemble performance
                optimized_prediction = np.zeros_like(actual_values)
                for name, weight in optimized_weights.items():
                    optimized_prediction += weight * model_predictions[name]
                
                optimized_mse = mean_squared_error(actual_values, optimized_prediction)
                optimized_r2 = r2_score(actual_values, optimized_prediction)
                
                result_dict = {
                    'optimized_weights': optimized_weights,
                    'optimized_mse': optimized_mse,
                    'optimized_r2': optimized_r2,
                    'model_performance': model_performance,
                    'optimization_success': True,
                    'optimization_message': result.message
                }
                
                logger.info(f"Weight optimization completed. R²: {optimized_r2:.4f}")
                return result_dict
            else:
                logger.warning(f"Weight optimization failed: {result.message}")
                return {
                    'optimized_weights': current_weights,
                    'optimization_success': False,
                    'optimization_message': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in performance-based weight optimization: {e}")
            return {'optimization_success': False, 'error': str(e)}

    def optimize_weights_sharpe(self, model_predictions: Dict[str, np.ndarray],
                               actual_values: np.ndarray,
                               returns: np.ndarray,
                               current_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Optimize weights to maximize Sharpe ratio"""
        try:
            logger.info("Optimizing weights to maximize Sharpe ratio")
            
            # Initialize weights if not provided
            if current_weights is None:
                n_models = len(model_predictions)
                current_weights = {name: 1.0 / n_models for name in model_predictions.keys()}
            
            # Calculate prediction returns for each model
            model_returns = {}
            for name, predictions in model_predictions.items():
                # Calculate directional accuracy and returns
                direction_correct = np.sign(np.diff(predictions)) == np.sign(np.diff(actual_values))
                model_returns[name] = returns * direction_correct
            
            # Define optimization objective (negative Sharpe ratio to minimize)
            def objective_function(weights_array):
                weights_dict = dict(zip(model_predictions.keys(), weights_array))
                
                # Calculate ensemble returns
                ensemble_returns = np.zeros_like(returns)
                for name, weight in weights_dict.items():
                    ensemble_returns += weight * model_returns[name]
                
                # Calculate Sharpe ratio
                mean_return = np.mean(ensemble_returns)
                std_return = np.std(ensemble_returns)
                
                if std_return > 0:
                    sharpe_ratio = mean_return / std_return
                else:
                    sharpe_ratio = 0
                
                return -sharpe_ratio  # Negative because we minimize
            
            # Define constraints
            n_models = len(model_predictions)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Define bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_models)]
            
            # Initial weights
            initial_weights = np.array(list(current_weights.values()))
            
            # Optimize weights
            result = minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimized_weights = dict(zip(model_predictions.keys(), result.x))
                
                # Calculate optimized Sharpe ratio
                optimized_returns = np.zeros_like(returns)
                for name, weight in optimized_weights.items():
                    optimized_returns += weight * model_returns[name]
                
                optimized_sharpe = -result.fun  # Convert back to positive
                
                result_dict = {
                    'optimized_weights': optimized_weights,
                    'optimized_sharpe': optimized_sharpe,
                    'optimization_success': True,
                    'optimization_message': result.message
                }
                
                logger.info(f"Sharpe ratio optimization completed. Sharpe: {optimized_sharpe:.4f}")
                return result_dict
            else:
                logger.warning(f"Sharpe ratio optimization failed: {result.message}")
                return {
                    'optimized_weights': current_weights,
                    'optimization_success': False,
                    'optimization_message': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in Sharpe ratio optimization: {e}")
            return {'optimization_success': False, 'error': str(e)}

    def optimize_weights_risk_parity(self, model_predictions: Dict[str, np.ndarray],
                                    actual_values: np.ndarray,
                                    current_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Optimize weights using risk parity approach"""
        try:
            logger.info("Optimizing weights using risk parity")
            
            # Initialize weights if not provided
            if current_weights is None:
                n_models = len(model_predictions)
                current_weights = {name: 1.0 / n_models for name in model_predictions.keys()}
            
            # Calculate prediction errors for each model
            model_errors = {}
            for name, predictions in model_predictions.items():
                errors = predictions - actual_values
                model_errors[name] = errors
            
            # Define risk parity objective
            def objective_function(weights_array):
                weights_dict = dict(zip(model_predictions.keys(), weights_array))
                
                # Calculate individual risk contributions
                risk_contributions = {}
                total_risk = 0
                
                for name, weight in weights_dict.items():
                    error_std = np.std(model_errors[name])
                    risk_contribution = weight * error_std
                    risk_contributions[name] = risk_contribution
                    total_risk += risk_contribution
                
                # Calculate risk parity deviation
                target_contribution = total_risk / len(weights_dict)
                parity_deviation = 0
                
                for contribution in risk_contributions.values():
                    parity_deviation += (contribution - target_contribution) ** 2
                
                return parity_deviation
            
            # Define constraints
            n_models = len(model_predictions)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Define bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_models)]
            
            # Initial weights
            initial_weights = np.array(list(current_weights.values()))
            
            # Optimize weights
            result = minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimized_weights = dict(zip(model_predictions.keys(), result.x))
                
                # Calculate risk contributions with optimized weights
                risk_contributions = {}
                total_risk = 0
                
                for name, weight in optimized_weights.items():
                    error_std = np.std(model_errors[name])
                    risk_contribution = weight * error_std
                    risk_contributions[name] = risk_contribution
                    total_risk += risk_contribution
                
                result_dict = {
                    'optimized_weights': optimized_weights,
                    'risk_contributions': risk_contributions,
                    'total_risk': total_risk,
                    'optimization_success': True,
                    'optimization_message': result.message
                }
                
                logger.info(f"Risk parity optimization completed. Total risk: {total_risk:.4f}")
                return result_dict
            else:
                logger.warning(f"Risk parity optimization failed: {result.message}")
                return {
                    'optimized_weights': current_weights,
                    'optimization_success': False,
                    'optimization_message': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return {'optimization_success': False, 'error': str(e)}

    def adaptive_weight_update(self, current_weights: Dict[str, float],
                             recent_performance: Dict[str, float],
                             market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Adaptively update weights based on recent performance and market conditions"""
        try:
            logger.info("Performing adaptive weight update")
            
            # Calculate performance-based adjustments
            total_performance = sum(max(perf, 0) for perf in recent_performance.values())
            
            if total_performance > 0:
                performance_weights = {
                    name: max(perf, 0) / total_performance 
                    for name, perf in recent_performance.items()
                }
            else:
                performance_weights = {
                    name: 1.0 / len(recent_performance) 
                    for name in recent_performance.keys()
                }
            
            # Calculate market condition adjustments
            market_adjustments = self._calculate_market_adjustments(market_conditions)
            
            # Combine current weights with performance and market adjustments
            updated_weights = {}
            for name in current_weights.keys():
                current_weight = current_weights[name]
                performance_weight = performance_weights.get(name, 0.5)
                market_adjustment = market_adjustments.get(name, 1.0)
                
                # Adaptive update formula
                new_weight = (
                    current_weight * (1 - self.learning_rate) +
                    performance_weight * self.learning_rate * market_adjustment
                )
                
                updated_weights[name] = max(self.min_weight, min(self.max_weight, new_weight))
            
            # Normalize weights to sum to 1
            total_weight = sum(updated_weights.values())
            if total_weight > 0:
                updated_weights = {
                    name: weight / total_weight 
                    for name, weight in updated_weights.items()
                }
            
            # Store weight update
            self.weights_history.append({
                'timestamp': datetime.now().isoformat(),
                'previous_weights': current_weights,
                'updated_weights': updated_weights,
                'recent_performance': recent_performance,
                'market_conditions': market_conditions
            })
            
            logger.info("Adaptive weight update completed")
            return updated_weights
            
        except Exception as e:
            logger.error(f"Error in adaptive weight update: {e}")
            return current_weights

    def _calculate_market_adjustments(self, market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate market condition-based adjustments for each model"""
        try:
            adjustments = {}
            
            # Extract market condition metrics
            volatility = market_conditions.get('volatility', 0.5)
            trend_strength = market_conditions.get('trend_strength', 0.5)
            market_regime = market_conditions.get('market_regime', 'normal')
            
            # Model-specific adjustments based on market conditions
            model_types = {
                'lstm': {'volatility_sensitivity': 1.2, 'trend_sensitivity': 1.1},
                'transformer': {'volatility_sensitivity': 1.0, 'trend_sensitivity': 1.3},
                'random_forest': {'volatility_sensitivity': 0.8, 'trend_sensitivity': 0.9},
                'gradient_boosting': {'volatility_sensitivity': 0.9, 'trend_sensitivity': 1.0},
                'lightgbm': {'volatility_sensitivity': 1.1, 'trend_sensitivity': 1.2},
                'xgboost': {'volatility_sensitivity': 1.0, 'trend_sensitivity': 1.1}
            }
            
            for model_name in self.weights_history[-1]['previous_weights'].keys() if self.weights_history else []:
                # Determine model type from name
                model_type = 'default'
                for type_name in model_types.keys():
                    if type_name in model_name.lower():
                        model_type = type_name
                        break
                
                if model_type in model_types:
                    sensitivities = model_types[model_type]
                    
                    # Calculate adjustment based on market conditions
                    volatility_adj = 1.0 + (volatility - 0.5) * (sensitivities['volatility_sensitivity'] - 1.0)
                    trend_adj = 1.0 + (trend_strength - 0.5) * (sensitivities['trend_sensitivity'] - 1.0)
                    
                    # Market regime adjustments
                    regime_adj = 1.0
                    if market_regime == 'high_volatility':
                        regime_adj = 0.8  # Reduce weights in high volatility
                    elif market_regime == 'trending':
                        regime_adj = 1.2  # Increase weights in trending markets
                    
                    # Combine adjustments
                    adjustments[model_name] = volatility_adj * trend_adj * regime_adj
                else:
                    adjustments[model_name] = 1.0
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating market adjustments: {e}")
            return {}

    def confidence_weighted_ensemble(self, model_predictions: Dict[str, np.ndarray],
                                   prediction_confidences: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate ensemble prediction using confidence-weighted averaging"""
        try:
            logger.info("Generating confidence-weighted ensemble prediction")
            
            # Normalize confidences
            total_confidence = sum(np.mean(conf) for conf in prediction_confidences.values())
            
            if total_confidence > 0:
                normalized_confidences = {
                    name: np.mean(conf) / total_confidence 
                    for name, conf in prediction_confidences.items()
                }
            else:
                n_models = len(model_predictions)
                normalized_confidences = {
                    name: 1.0 / n_models 
                    for name in model_predictions.keys()
                }
            
            # Generate weighted ensemble prediction
            ensemble_prediction = np.zeros_like(list(model_predictions.values())[0])
            
            for name, predictions in model_predictions.items():
                weight = normalized_confidences[name]
                ensemble_prediction += weight * predictions
            
            logger.info("Confidence-weighted ensemble prediction generated")
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"Error in confidence-weighted ensemble: {e}")
            return np.zeros_like(list(model_predictions.values())[0])

    def get_weight_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get weight history for the specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_history = [
                entry for entry in self.weights_history
                if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
            ]
            return recent_history
        except Exception as e:
            logger.error(f"Error getting weight history: {e}")
            return []

    def get_weight_statistics(self) -> Dict[str, Any]:
        """Get weight change statistics"""
        try:
            if len(self.weights_history) < 2:
                return {'total_updates': 0}
            
            # Calculate weight change statistics
            weight_changes = []
            for i in range(1, len(self.weights_history)):
                prev_weights = self.weights_history[i-1]['updated_weights']
                curr_weights = self.weights_history[i]['updated_weights']
                
                for model_name in prev_weights.keys():
                    if model_name in curr_weights:
                        change = curr_weights[model_name] - prev_weights[model_name]
                        weight_changes.append(change)
            
            if weight_changes:
                return {
                    'total_updates': len(self.weights_history),
                    'average_weight_change': np.mean(weight_changes),
                    'weight_change_std': np.std(weight_changes),
                    'max_weight_change': np.max(weight_changes),
                    'min_weight_change': np.min(weight_changes),
                    'last_update': self.weights_history[-1]['timestamp']
                }
            else:
                return {'total_updates': len(self.weights_history)}
                
        except Exception as e:
            logger.error(f"Error getting weight statistics: {e}")
            return {'total_updates': 0}


# Example usage
if __name__ == "__main__":
    config = {
        'learning_rate': 0.01,
        'adaptation_window': 100,
        'confidence_threshold': 0.7,
        'optimization_method': 'sharpe',
        'min_weight': 0.01,
        'max_weight': 0.5
    }
    
    dynamic_weighting = DynamicWeighting(config)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    model_predictions = {
        'model1': np.random.randn(n_samples),
        'model2': np.random.randn(n_samples),
        'model3': np.random.randn(n_samples)
    }
    
    actual_values = np.random.randn(n_samples)
    returns = np.random.randn(n_samples) * 0.01
    
    # Test different optimization methods
    results_performance = dynamic_weighting.optimize_weights_performance(
        model_predictions, actual_values
    )
    
    results_sharpe = dynamic_weighting.optimize_weights_sharpe(
        model_predictions, actual_values, returns
    )
    
    results_risk_parity = dynamic_weighting.optimize_weights_risk_parity(
        model_predictions, actual_values
    )
    
    print("Performance optimization:", results_performance['optimized_r2'])
    print("Sharpe optimization:", results_sharpe['optimized_sharpe'])
    print("Risk parity optimization:", results_risk_parity['total_risk']) 