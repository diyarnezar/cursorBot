#!/usr/bin/env python3
"""
ULTRA-ADVANCED Parameter Optimizer Module
Autonomous parameter adjustment and continuous learning system
"""

import logging
import numpy as np
import pandas as pd
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import deque
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os

class ParameterOptimizer:
    """
    ULTRA-ADVANCED Parameter Optimizer with maximum intelligence:
    
    Features:
    - Autonomous parameter adjustment based on performance
    - Dynamic hyperparameter tuning with Optuna
    - Performance-based optimization with multi-objective functions
    - Market regime adaptation
    - Continuous learning system
    - Real-time parameter updates
    - Multi-parameter optimization
    - Adaptive optimization strategies
    """
    
    def __init__(self, 
                 optimization_interval: int = 3600,  # 1 hour
                 max_trials: int = 100,
                 enable_continuous: bool = True,
                 performance_threshold: float = 0.01):
        """
        Initialize the Parameter Optimizer.
        
        Args:
            optimization_interval: Interval between optimizations in seconds
            max_trials: Maximum trials for hyperparameter optimization
            enable_continuous: Whether to enable continuous optimization
            performance_threshold: Minimum performance improvement threshold
        """
        self.optimization_interval = optimization_interval
        self.max_trials = max_trials
        self.enable_continuous = enable_continuous
        self.performance_threshold = performance_threshold
        
        # Parameter storage
        self.current_parameters = {}
        self.parameter_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        
        # Optimization state
        self.is_optimizing = False
        self.optimization_thread = None
        self.last_optimization = datetime.now()
        
        # Study storage
        self.studies = {}
        
        # Performance tracking
        self.baseline_performance = None
        self.best_performance = None
        self.optimization_count = 0
        
        # Market regime tracking
        self.current_regime = 'NORMAL'
        self.regime_parameters = {}
        
        # Load existing parameters
        self._load_parameters()
        
        # Start continuous optimization if enabled
        if self.enable_continuous:
            self._start_continuous_optimization()
        
        logging.info("ULTRA-ADVANCED Parameter Optimizer initialized.")
    
    def optimize_parameters(self, 
                          parameter_space: Dict[str, Any],
                          objective_function: Callable,
                          optimization_type: str = 'single_objective') -> Dict[str, Any]:
        """
        Optimize parameters using advanced optimization techniques.
        
        Args:
            parameter_space: Dictionary defining parameter search space
            objective_function: Function to optimize
            optimization_type: Type of optimization ('single_objective', 'multi_objective')
            
        Returns:
            Dictionary with optimized parameters and results
        """
        try:
            logging.info(f"Starting parameter optimization: {optimization_type}")
            
            # Create or load study
            study_name = f"optimization_{optimization_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            study = optuna.create_study(
                direction='minimize' if optimization_type == 'single_objective' else 'maximize',
                sampler=TPESampler(seed=42)
            )
            
            # Run optimization
            if optimization_type == 'single_objective':
                result = self._run_single_objective_optimization(study, parameter_space, objective_function)
            else:
                result = self._run_multi_objective_optimization(study, parameter_space, objective_function)
            
            # Store study
            self.studies[study_name] = study
            
            # Update current parameters
            self._update_parameters(result['best_parameters'])
            
            # Track performance
            self._track_optimization_performance(result)
            
            logging.info(f"Parameter optimization completed. Best score: {result['best_score']}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in parameter optimization: {e}")
            return {}
    
    def _run_single_objective_optimization(self, study: optuna.Study, 
                                         parameter_space: Dict[str, Any],
                                         objective_function: Callable) -> Dict[str, Any]:
        """Run single-objective optimization."""
        try:
            def objective(trial):
                # Sample parameters from search space
                params = self._sample_parameters(trial, parameter_space)
                
                # Evaluate objective function
                score = objective_function(params)
                
                return score
            
            # Optimize
            study.optimize(objective, n_trials=self.max_trials, timeout=3600)
            
            # Get best results
            best_params = study.best_params
            best_score = study.best_value
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'optimization_type': 'single_objective',
                'n_trials': len(study.trials),
                'study': study
            }
            
        except Exception as e:
            logging.error(f"Error in single-objective optimization: {e}")
            return {}
    
    def _run_multi_objective_optimization(self, study: optuna.Study,
                                        parameter_space: Dict[str, Any],
                                        objective_function: Callable) -> Dict[str, Any]:
        """Run multi-objective optimization."""
        try:
            def objective(trial):
                # Sample parameters from search space
                params = self._sample_parameters(trial, parameter_space)
                
                # Evaluate objective function (should return multiple objectives)
                scores = objective_function(params)
                
                # Return weighted sum for now (could be extended to true multi-objective)
                if isinstance(scores, (list, tuple)):
                    weights = [0.4, 0.3, 0.3]  # Weights for different objectives
                    weighted_score = sum(s * w for s, w in zip(scores, weights))
                    return weighted_score
                else:
                    return scores
            
            # Optimize
            study.optimize(objective, n_trials=self.max_trials, timeout=3600)
            
            # Get best results
            best_params = study.best_params
            best_score = study.best_value
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'optimization_type': 'multi_objective',
                'n_trials': len(study.trials),
                'study': study
            }
            
        except Exception as e:
            logging.error(f"Error in multi-objective optimization: {e}")
            return {}
    
    def _sample_parameters(self, trial: optuna.Trial, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from the search space."""
        try:
            params = {}
            
            for param_name, param_config in parameter_space.items():
                param_type = param_config.get('type', 'float')
                
                if param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['min'],
                        param_config['max'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['min'],
                        param_config['max'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
                elif param_type == 'uniform':
                    params[param_name] = trial.suggest_uniform(
                        param_name,
                        param_config['min'],
                        param_config['max']
                    )
            
            return params
            
        except Exception as e:
            logging.error(f"Error sampling parameters: {e}")
            return {}
    
    def adaptive_parameter_adjustment(self, 
                                    current_performance: float,
                                    target_performance: float,
                                    parameter_sensitivity: Dict[str, float]) -> Dict[str, Any]:
        """
        Adaptively adjust parameters based on current performance.
        
        Args:
            current_performance: Current performance metric
            target_performance: Target performance metric
            parameter_sensitivity: Dictionary of parameter sensitivity scores
            
        Returns:
            Dictionary with adjusted parameters
        """
        try:
            # Calculate performance gap
            performance_gap = target_performance - current_performance
            
            # Determine adjustment direction
            if performance_gap > self.performance_threshold:
                adjustment_direction = 'increase'
            elif performance_gap < -self.performance_threshold:
                adjustment_direction = 'decrease'
            else:
                return self.current_parameters  # No adjustment needed
            
            # Calculate adjustment magnitude
            adjustment_magnitude = min(abs(performance_gap) * 0.1, 0.2)  # Max 20% adjustment
            
            # Adjust parameters based on sensitivity
            adjusted_parameters = {}
            for param_name, current_value in self.current_parameters.items():
                sensitivity = parameter_sensitivity.get(param_name, 0.5)
                
                if isinstance(current_value, (int, float)):
                    if adjustment_direction == 'increase':
                        adjustment = current_value * adjustment_magnitude * sensitivity
                        adjusted_parameters[param_name] = current_value + adjustment
                    else:
                        adjustment = current_value * adjustment_magnitude * sensitivity
                        adjusted_parameters[param_name] = current_value - adjustment
                else:
                    adjusted_parameters[param_name] = current_value
            
            # Apply regime-specific adjustments
            adjusted_parameters = self._apply_regime_adjustments(adjusted_parameters)
            
            # Update parameters
            self._update_parameters(adjusted_parameters)
            
            logging.info(f"Adaptive parameter adjustment applied. Performance gap: {performance_gap:.4f}")
            
            return adjusted_parameters
            
        except Exception as e:
            logging.error(f"Error in adaptive parameter adjustment: {e}")
            return self.current_parameters
    
    def _apply_regime_adjustments(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply market regime-specific parameter adjustments."""
        try:
            adjusted_parameters = parameters.copy()
            
            # Get regime-specific adjustments
            regime_adjustments = self.regime_parameters.get(self.current_regime, {})
            
            for param_name, adjustment in regime_adjustments.items():
                if param_name in adjusted_parameters:
                    current_value = adjusted_parameters[param_name]
                    
                    if isinstance(current_value, (int, float)) and isinstance(adjustment, (int, float)):
                        adjusted_parameters[param_name] = current_value * (1 + adjustment)
            
            return adjusted_parameters
            
        except Exception as e:
            logging.error(f"Error applying regime adjustments: {e}")
            return parameters
    
    def optimize_for_regime(self, regime: str, performance_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize parameters for a specific market regime.
        
        Args:
            regime: Market regime name
            performance_data: Historical performance data for the regime
            
        Returns:
            Dictionary with regime-optimized parameters
        """
        try:
            logging.info(f"Optimizing parameters for regime: {regime}")
            
            # Define regime-specific parameter space
            regime_parameter_space = self._get_regime_parameter_space(regime)
            
            # Define objective function for regime
            def regime_objective(params):
                return self._evaluate_regime_performance(params, performance_data)
            
            # Run optimization
            result = self.optimize_parameters(regime_parameter_space, regime_objective)
            
            # Store regime parameters
            if result:
                self.regime_parameters[regime] = result['best_parameters']
            
            return result
            
        except Exception as e:
            logging.error(f"Error optimizing for regime: {e}")
            return {}
    
    def _get_regime_parameter_space(self, regime: str) -> Dict[str, Any]:
        """Get parameter search space for a specific regime."""
        try:
            # Define different parameter spaces for different regimes
            regime_spaces = {
                'HIGH_VOLATILITY': {
                    'risk_multiplier': {'type': 'float', 'min': 0.5, 'max': 0.8},
                    'position_size_multiplier': {'type': 'float', 'min': 0.6, 'max': 0.9},
                    'stop_loss_multiplier': {'type': 'float', 'min': 1.2, 'max': 1.8},
                    'confidence_threshold': {'type': 'float', 'min': 0.7, 'max': 0.9}
                },
                'LOW_VOLATILITY': {
                    'risk_multiplier': {'type': 'float', 'min': 1.0, 'max': 1.5},
                    'position_size_multiplier': {'type': 'float', 'min': 1.1, 'max': 1.4},
                    'stop_loss_multiplier': {'type': 'float', 'min': 0.8, 'max': 1.2},
                    'confidence_threshold': {'type': 'float', 'min': 0.5, 'max': 0.7}
                },
                'TRENDING': {
                    'risk_multiplier': {'type': 'float', 'min': 0.8, 'max': 1.2},
                    'position_size_multiplier': {'type': 'float', 'min': 1.0, 'max': 1.3},
                    'stop_loss_multiplier': {'type': 'float', 'min': 1.0, 'max': 1.5},
                    'confidence_threshold': {'type': 'float', 'min': 0.6, 'max': 0.8}
                },
                'SIDEWAYS': {
                    'risk_multiplier': {'type': 'float', 'min': 0.6, 'max': 1.0},
                    'position_size_multiplier': {'type': 'float', 'min': 0.8, 'max': 1.1},
                    'stop_loss_multiplier': {'type': 'float', 'min': 0.9, 'max': 1.3},
                    'confidence_threshold': {'type': 'float', 'min': 0.7, 'max': 0.9}
                }
            }
            
            return regime_spaces.get(regime, regime_spaces['NORMAL'])
            
        except Exception as e:
            logging.error(f"Error getting regime parameter space: {e}")
            return {}
    
    def _evaluate_regime_performance(self, params: Dict[str, Any], 
                                   performance_data: pd.DataFrame) -> float:
        """Evaluate performance for a specific regime."""
        try:
            # This would implement actual performance evaluation
            # For now, return a simple metric
            if 'risk_multiplier' in params and 'position_size_multiplier' in params:
                # Simple scoring based on parameter values
                risk_score = 1.0 - abs(params['risk_multiplier'] - 1.0)
                position_score = 1.0 - abs(params['position_size_multiplier'] - 1.0)
                return -(risk_score + position_score)  # Negative for minimization
            
            return 0.0
            
        except Exception as e:
            logging.error(f"Error evaluating regime performance: {e}")
            return 0.0
    
    def continuous_learning_update(self, new_performance: float, 
                                 new_parameters: Dict[str, Any]) -> None:
        """
        Update the continuous learning system with new performance data.
        
        Args:
            new_performance: New performance metric
            new_parameters: Parameters that produced this performance
        """
        try:
            # Store performance and parameters
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'performance': new_performance,
                'parameters': new_parameters.copy()
            })
            
            # Update best performance
            if self.best_performance is None or new_performance > self.best_performance:
                self.best_performance = new_performance
                logging.info(f"New best performance achieved: {new_performance:.4f}")
            
            # Check if optimization is needed
            if self._should_trigger_optimization():
                self._trigger_optimization()
            
        except Exception as e:
            logging.error(f"Error in continuous learning update: {e}")
    
    def _should_trigger_optimization(self) -> bool:
        """Check if optimization should be triggered."""
        try:
            # Check time-based trigger
            time_since_last = datetime.now() - self.last_optimization
            if time_since_last.total_seconds() < self.optimization_interval:
                return False
            
            # Check performance-based trigger
            if len(self.performance_history) < 10:
                return False
            
            recent_performances = [p['performance'] for p in list(self.performance_history)[-10:]]
            avg_recent_performance = np.mean(recent_performances)
            
            if self.best_performance and avg_recent_performance < self.best_performance * 0.95:
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking optimization trigger: {e}")
            return False
    
    def _trigger_optimization(self) -> None:
        """Trigger parameter optimization."""
        try:
            if not self.is_optimizing:
                self.is_optimizing = True
                
                # Start optimization in separate thread
                optimization_thread = threading.Thread(target=self._run_background_optimization)
                optimization_thread.daemon = True
                optimization_thread.start()
                
                logging.info("Background optimization triggered")
            
        except Exception as e:
            logging.error(f"Error triggering optimization: {e}")
    
    def _run_background_optimization(self) -> None:
        """Run background optimization."""
        try:
            # Define parameter space for background optimization
            parameter_space = {
                'risk_multiplier': {'type': 'float', 'min': 0.5, 'max': 1.5},
                'position_size_multiplier': {'type': 'float', 'min': 0.8, 'max': 1.4},
                'stop_loss_multiplier': {'type': 'float', 'min': 0.8, 'max': 1.8},
                'confidence_threshold': {'type': 'float', 'min': 0.5, 'max': 0.9}
            }
            
            # Define objective function
            def background_objective(params):
                return self._evaluate_background_performance(params)
            
            # Run optimization
            result = self.optimize_parameters(parameter_space, background_objective)
            
            # Update parameters if improvement found
            if result and result.get('best_score', 0) > self.best_performance:
                self._update_parameters(result['best_parameters'])
                logging.info("Background optimization completed with improvement")
            
            self.is_optimizing = False
            self.last_optimization = datetime.now()
            
        except Exception as e:
            logging.error(f"Error in background optimization: {e}")
            self.is_optimizing = False
    
    def _evaluate_background_performance(self, params: Dict[str, Any]) -> float:
        """Evaluate performance for background optimization."""
        try:
            # Use recent performance history to evaluate parameters
            if len(self.performance_history) < 5:
                return 0.0
            
            # Simple evaluation based on parameter similarity to recent good performers
            recent_good_performances = sorted(
                self.performance_history, 
                key=lambda x: x['performance'], 
                reverse=True
            )[:5]
            
            total_score = 0.0
            for good_perf in recent_good_performances:
                similarity = self._calculate_parameter_similarity(params, good_perf['parameters'])
                total_score += similarity * good_perf['performance']
            
            return -total_score  # Negative for minimization
            
        except Exception as e:
            logging.error(f"Error evaluating background performance: {e}")
            return 0.0
    
    def _calculate_parameter_similarity(self, params1: Dict[str, Any], 
                                      params2: Dict[str, Any]) -> float:
        """Calculate similarity between two parameter sets."""
        try:
            if not params1 or not params2:
                return 0.0
            
            similarities = []
            for key in set(params1.keys()) & set(params2.keys()):
                if isinstance(params1[key], (int, float)) and isinstance(params2[key], (int, float)):
                    # Calculate normalized difference
                    max_val = max(abs(params1[key]), abs(params2[key]))
                    if max_val > 0:
                        diff = abs(params1[key] - params2[key]) / max_val
                        similarities.append(1.0 - diff)
                    else:
                        similarities.append(1.0)
                else:
                    # For non-numeric parameters, check equality
                    similarities.append(1.0 if params1[key] == params2[key] else 0.0)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating parameter similarity: {e}")
            return 0.0
    
    def _update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """Update current parameters."""
        try:
            # Store parameter history
            self.parameter_history.append({
                'timestamp': datetime.now().isoformat(),
                'parameters': self.current_parameters.copy()
            })
            
            # Update current parameters
            self.current_parameters.update(new_parameters)
            
            # Save parameters
            self._save_parameters()
            
            logging.info(f"Parameters updated: {list(new_parameters.keys())}")
            
        except Exception as e:
            logging.error(f"Error updating parameters: {e}")
    
    def _load_parameters(self) -> None:
        """Load parameters from file."""
        try:
            param_file = 'parameters/optimized_parameters.json'
            if os.path.exists(param_file):
                with open(param_file, 'r') as f:
                    self.current_parameters = json.load(f)
                logging.info("Parameters loaded from file")
            else:
                # Set default parameters
                self.current_parameters = {
                    'risk_multiplier': 1.0,
                    'position_size_multiplier': 1.0,
                    'stop_loss_multiplier': 1.0,
                    'confidence_threshold': 0.7
                }
                logging.info("Default parameters set")
                
        except Exception as e:
            logging.error(f"Error loading parameters: {e}")
            self.current_parameters = {}
    
    def _save_parameters(self) -> None:
        """Save parameters to file."""
        try:
            os.makedirs('parameters', exist_ok=True)
            param_file = 'parameters/optimized_parameters.json'
            
            with open(param_file, 'w') as f:
                json.dump(self.current_parameters, f, indent=2)
            
            logging.info("Parameters saved to file")
            
        except Exception as e:
            logging.error(f"Error saving parameters: {e}")
    
    def _start_continuous_optimization(self) -> None:
        """Start continuous optimization loop."""
        try:
            self.optimization_thread = threading.Thread(target=self._continuous_optimization_loop)
            self.optimization_thread.daemon = True
            self.optimization_thread.start()
            
            logging.info("Continuous optimization started")
            
        except Exception as e:
            logging.error(f"Error starting continuous optimization: {e}")
    
    def _continuous_optimization_loop(self) -> None:
        """Continuous optimization loop."""
        while self.enable_continuous:
            try:
                time.sleep(self.optimization_interval)
                
                if self._should_trigger_optimization():
                    self._trigger_optimization()
                    
            except Exception as e:
                logging.error(f"Error in continuous optimization loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        try:
            summary = {
                'current_parameters': self.current_parameters,
                'best_performance': self.best_performance,
                'optimization_count': self.optimization_count,
                'regime_parameters': self.regime_parameters,
                'is_optimizing': self.is_optimizing,
                'last_optimization': self.last_optimization.isoformat(),
                'performance_history_length': len(self.performance_history),
                'parameter_history_length': len(self.parameter_history)
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error getting optimization summary: {e}")
            return {} 